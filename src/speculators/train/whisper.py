"""Experimental in-process Whisper drafter training building blocks.

Checkpoints use explicit helpers here, not the generic vLLM/HF draft loader.
All frozen teacher-owned draft tensors are saved for independent reloading.
"""

import copy
import json
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, WhisperForConditionalGeneration

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.models.eagle3.whisper import WhisperEagle3DraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.whisper_eagle3 import eagle3_whisper_loss


@torch.no_grad()
def generate_whisper_tokens(
    teacher, audio, prompt, *, max_new_tokens, attention_mask=None
):
    """Keep the decoder prompt and EOS which Whisper strips from plain output."""
    if attention_mask is None:
        # Short-form features are fixed-length encoder inputs with no decoder padding.
        attention_mask = torch.ones(
            audio.shape[0], audio.shape[-1], device=audio.device, dtype=torch.long
        )
    tokens = teacher.generate(
        input_features=audio,
        attention_mask=attention_mask,
        decoder_input_ids=prompt,
        do_sample=False,
        num_beams=1,
        return_timestamps=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
    ).sequences.to(teacher.device)
    if not torch.equal(tokens[:, : prompt.shape[1]], prompt):
        raise RuntimeError("Whisper generation did not preserve the decoder prompt")
    return tokens


def build_whisper_draft(
    teacher: WhisperForConditionalGeneration,
    target_layer_ids: list[int],
    *,
    block_size: int = 4,
    num_layers: int = 1,
    attention_implementation: str = "eager",
    algorithm: str = "dflash",
) -> DFlashDraftModel | WhisperEagle3DraftModel:
    """Build a native drafter with Whisper's dimensions and verifier weights."""
    if algorithm not in ("dflash", "eagle3"):
        raise ValueError(f"Unsupported Whisper drafter: {algorithm}")
    if block_size < 2 or num_layers < 1:  # noqa: PLR2004
        raise ValueError("Need block_size >= 2 and num_layers >= 1")
    cfg = teacher.config
    head_dim = cfg.d_model // cfg.decoder_attention_heads
    if head_dim % 2:
        raise ValueError("Draft rotary attention requires an even head dimension")
    transformer = Qwen3Config(
        hidden_size=cfg.d_model,
        intermediate_size=cfg.decoder_ffn_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=cfg.decoder_attention_heads,
        num_key_value_heads=cfg.decoder_attention_heads,
        head_dim=head_dim,
        vocab_size=cfg.vocab_size,
        max_position_embeddings=cfg.max_target_positions,
        _attn_implementation=attention_implementation,
    )
    if algorithm == "dflash":
        draft = DFlashDraftModel(
            DFlashSpeculatorConfig(
                transformer_layer_config=transformer,
                draft_vocab_size=cfg.vocab_size,
                block_size=block_size,
                aux_hidden_state_layer_ids=target_layer_ids,
                mask_token_id=cfg.pad_token_id,
                speculators_config=SpeculatorsConfig(
                    algorithm="dflash",
                    proposal_methods=[
                        GreedyTokenProposalConfig(speculative_tokens=block_size - 1)
                    ],
                    default_proposal_method="greedy",
                    verifier=VerifierConfig.from_config(cfg, name_or_path=None),
                ),
            )
        )
    else:
        draft = WhisperEagle3DraftModel(
            Eagle3SpeculatorConfig(
                transformer_layer_config=transformer,
                draft_vocab_size=cfg.vocab_size,
                eagle_aux_hidden_state_layer_ids=target_layer_ids,
                norm_before_fc=True,
                norm_output=True,
                speculators_config=SpeculatorsConfig(
                    algorithm="eagle3",
                    proposal_methods=[
                        GreedyTokenProposalConfig(speculative_tokens=block_size - 1)
                    ],
                    default_proposal_method="greedy",
                    verifier=VerifierConfig.from_config(cfg, name_or_path=None),
                ),
            ),
            block_size=block_size,
        )
    draft.to(device=teacher.device, dtype=teacher.dtype)
    draft.verifier_norm = copy.deepcopy(teacher.model.decoder.layer_norm)
    with torch.no_grad():
        draft.embed_tokens.weight.copy_(teacher.model.decoder.embed_tokens.weight)
        draft.lm_head.weight.copy_(teacher.proj_out.weight)
        draft.verifier_lm_head.weight.copy_(teacher.proj_out.weight)
    for module in (
        draft.embed_tokens,
        draft.verifier_lm_head,
        draft.verifier_norm,
    ):
        module.requires_grad_(False)
    if algorithm == "dflash":
        draft.lm_head.requires_grad_(False)
    return draft


@dataclass(frozen=True)
class WhisperLossOptions:
    implementation: str = "eager"
    policy_targets: bool = True
    response_ce_weight: float = 0.0
    position_weight: str = "fixed-exp-decay"
    gamma: float = 4.0
    dpace_alpha: float = 0.5
    rollout_decay: float = 1.0


def whisper_draft_loss(
    draft, features, *, max_anchors=32, options=None, suppressed_tokens=()
):
    """Policy-aligned distillation with independent partial blocks per utterance."""
    options = options or WhisperLossOptions()
    if isinstance(draft, WhisperEagle3DraftModel):
        return eagle3_whisper_loss(
            draft, features, options=options, suppressed_tokens=suppressed_tokens
        )
    if "document_ids" in features:
        max_anchors *= torch.unique(features["document_ids"]).numel()
    _, logits, targets, mask, indices = draft._backbone_forward(  # noqa: SLF001 -- reuse native anchored transformer
        **features,
        max_anchors=max_anchors,
        allow_partial_blocks=True,
        anchors_per_document=True,
    )
    raw_agreement = targets.argmax(-1) == features["input_ids"][:, indices]
    suppressed_tokens = [
        token for token in suppressed_tokens if 0 <= token < logits.shape[-1]
    ]
    if options.policy_targets and suppressed_tokens:
        # Finite sentinels avoid 0 * inf NaNs in KL/fused kernels.
        logits = logits.index_fill(
            -1,
            torch.as_tensor(suppressed_tokens, device=logits.device, dtype=torch.long),
            torch.finfo(logits.dtype).min,
        )
        targets = targets.index_fill(
            -1,
            torch.as_tensor(suppressed_tokens, device=targets.device, dtype=torch.long),
            torch.finfo(targets.dtype).min,
        )
    losses = resolve_loss_config("kl_div", implementation=options.implementation)
    if options.response_ce_weight:
        labels = features["input_ids"][:, indices]

        def response_ce(scores, _targets):
            return torch.nn.functional.cross_entropy(
                scores.float().reshape(-1, scores.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape_as(labels)

        losses["response_ce"] = (response_ce, options.response_ce_weight)
    loss, metrics = compute_metrics(
        logits,
        targets,
        mask,
        draft.block_size,
        gamma=options.gamma,
        loss_config=losses,
        per_position_loss_weight=options.position_weight,
        dpace_alpha=options.dpace_alpha,
    )
    count = mask.sum()
    metrics["weighted_loss_sum"] = loss.detach() * count
    metrics["weighted_loss_total"] = count
    metrics["raw_teacher_agreement_sum"] = (raw_agreement * mask).sum()
    metrics["raw_teacher_agreement_total"] = count
    metrics["policy_teacher_agreement_sum"] = (
        (targets.argmax(-1) == features["input_ids"][:, indices]) * mask
    ).sum()
    metrics["policy_teacher_agreement_total"] = count
    return loss, metrics


def make_whisper_loss(
    draft, *, options, max_anchors, suppressed_tokens=(), compile_model=False
):
    function = partial(
        whisper_draft_loss,
        draft,
        options=options,
        max_anchors=max_anchors,
        suppressed_tokens=suppressed_tokens,
    )
    if compile_model:
        return torch.compile(function, dynamic=True)
    return function


def metrics_to_cpu(metrics):
    values = (
        torch.stack([value.detach().float() for value in metrics.values()])
        .cpu()
        .tolist()
    )
    return dict(zip(metrics, values, strict=True))


def train_whisper_step(draft, optimizer, features, *, max_anchors=4, metrics_sink=None):
    """Compatibility single-step trainer; production batches use the shared loss."""
    draft.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.compiler.set_stance("force_eager"):
        loss, metrics = whisper_draft_loss(draft, features, max_anchors=max_anchors)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite Whisper draft loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in draft.parameters() if p.requires_grad],
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    optimizer.step()
    if metrics_sink is not None:
        metrics_sink.update(metrics_to_cpu(metrics))
    return float(loss.detach())


def whisper_autocast(device, dtype):
    device = torch.device(device)
    return (
        torch.autocast(device.type, dtype=dtype)
        if device.type in ("cuda", "cpu") and dtype != torch.float32
        else nullcontext()
    )


def save_whisper_draft(
    draft: DFlashDraftModel | WhisperEagle3DraftModel, directory: Path
) -> None:
    """Save a self-contained experimental training checkpoint (no features)."""
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "whisper-eagle3-experimental-v1"
        if isinstance(draft, WhisperEagle3DraftModel)
        else "whisper-dflash-experimental-v1",
        "block_size": draft.block_size,
        "draft_config": draft.config.to_dict(),
        "verifier_norm_eps": draft.verifier_norm.eps,
        "attention_implementation": draft._attn_impl,  # noqa: SLF001 -- persisted backend
    }
    (directory / "whisper_draft.json").write_text(json.dumps(metadata, indent=2))
    save_file(
        {
            name: value.detach().cpu().contiguous().clone()
            for name, value in draft.state_dict().items()
        },
        str(directory / "draft.safetensors"),
    )


def load_whisper_draft(directory: Path) -> DFlashDraftModel | WhisperEagle3DraftModel:
    """Reload without downloading or instantiating a Whisper teacher."""
    metadata = json.loads((directory / "whisper_draft.json").read_text())
    formats = {
        "whisper-dflash-experimental-v1": DFlashSpeculatorConfig,
        "whisper-eagle3-experimental-v1": Eagle3SpeculatorConfig,
    }
    if metadata["format"] not in formats:
        raise ValueError("Unsupported Whisper draft checkpoint format")
    config = formats[metadata["format"]].model_validate(metadata["draft_config"])
    # HF serialization omits the private attention setting; restore the saved
    # backend, with eager as the backwards-compatible default.
    config.transformer_layer_config._attn_implementation = metadata.get(  # noqa: SLF001
        "attention_implementation", "eager"
    )
    draft = (
        WhisperEagle3DraftModel(config, block_size=metadata["block_size"])
        if isinstance(config, Eagle3SpeculatorConfig)
        else DFlashDraftModel(config)
    )
    draft.verifier_norm = torch.nn.LayerNorm(
        draft.hidden_size, eps=metadata["verifier_norm_eps"]
    )
    draft.load_state_dict(load_file(str(directory / "draft.safetensors")))
    for module in (
        draft.embed_tokens,
        draft.verifier_lm_head,
        draft.verifier_norm,
    ):
        module.requires_grad_(False)
    if isinstance(draft, DFlashDraftModel):
        draft.lm_head.requires_grad_(False)
    return draft
