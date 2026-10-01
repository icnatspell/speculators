"""Experimental in-process Whisper/DFlash training building blocks.

Checkpoints use explicit helpers here, not the generic vLLM/HF draft loader.
All frozen teacher-owned draft tensors are saved for independent reloading.
"""

import copy
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, WhisperForConditionalGeneration

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import resolve_loss_config
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig


@torch.no_grad()
def generate_whisper_tokens(teacher, audio, prompt, *, max_new_tokens):
    """Keep the decoder prompt and EOS which Whisper strips from plain output."""
    tokens = teacher.generate(
        input_features=audio,
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
) -> DFlashDraftModel:
    """Build a Qwen3 draft with Whisper's width/vocabulary and frozen head."""
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
        _attn_implementation="eager",
    )
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
    draft.to(device=teacher.device, dtype=teacher.dtype)
    draft.verifier_norm = copy.deepcopy(teacher.model.decoder.layer_norm)
    with torch.no_grad():
        draft.embed_tokens.weight.copy_(teacher.model.decoder.embed_tokens.weight)
        draft.lm_head.weight.copy_(teacher.proj_out.weight)
        draft.verifier_lm_head.weight.copy_(teacher.proj_out.weight)
    for module in (
        draft.embed_tokens,
        draft.lm_head,
        draft.verifier_lm_head,
        draft.verifier_norm,
    ):
        module.requires_grad_(False)
    return draft


def train_whisper_step(
    draft: DFlashDraftModel,
    optimizer: torch.optim.Optimizer,
    features: dict[str, torch.Tensor],
    *,
    max_anchors: int = 4,
    metrics_sink: dict | None = None,
) -> float:
    """One finite-gradient update; caller owns the current feature batch."""
    valid = features["loss_mask"][0, : -draft.block_size].bool()
    if not valid.any():
        raise ValueError("Continuation is too short for an anchored draft block")
    draft.train()
    optimizer.zero_grad(set_to_none=True)
    # Existing DFlash wrappers compile whenever a GPU is present, even for CPU
    # tensors. This experimental path deliberately uses eager attention/loss.
    with torch.compiler.set_stance("force_eager"):
        _, loss, metrics = draft(
            **features,
            max_anchors=max_anchors,
            loss_config=resolve_loss_config("kl_div", implementation="eager"),
        )
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
        values = (
            torch.stack([value.detach().float() for value in metrics.values()])
            .cpu()
            .tolist()
        )
        metrics_sink.update(zip(metrics, values, strict=True))
    return float(loss.detach())


def save_whisper_draft(draft: DFlashDraftModel, directory: Path) -> None:
    """Save a self-contained experimental training checkpoint (no features)."""
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "whisper-dflash-experimental-v1",
        "draft_config": draft.config.to_dict(),
        "verifier_norm_eps": draft.verifier_norm.eps,
    }
    (directory / "whisper_draft.json").write_text(json.dumps(metadata, indent=2))
    save_file(
        {
            name: value.detach().cpu().contiguous().clone()
            for name, value in draft.state_dict().items()
        },
        str(directory / "draft.safetensors"),
    )


def load_whisper_draft(directory: Path) -> DFlashDraftModel:
    """Reload without downloading or instantiating a Whisper teacher."""
    metadata = json.loads((directory / "whisper_draft.json").read_text())
    if metadata["format"] != "whisper-dflash-experimental-v1":
        raise ValueError("Unsupported Whisper draft checkpoint format")
    config = DFlashSpeculatorConfig.model_validate(metadata["draft_config"])
    # HF config serialization omits the private attention setting. This
    # experimental checkpoint format always uses eager attention.
    config.transformer_layer_config._attn_implementation = "eager"  # noqa: SLF001
    draft = DFlashDraftModel(config)
    draft.verifier_norm = torch.nn.LayerNorm(
        draft.hidden_size, eps=metadata["verifier_norm_eps"]
    )
    draft.load_state_dict(load_file(str(directory / "draft.safetensors")))
    for module in (
        draft.embed_tokens,
        draft.lm_head,
        draft.verifier_lm_head,
        draft.verifier_norm,
    ):
        module.requires_grad_(False)
    return draft
