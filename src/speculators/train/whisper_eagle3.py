"""Whisper token alignment around native EAGLE-3 multi-step training."""

from collections import defaultdict

import torch

from speculators.losses import resolve_loss_config
from speculators.models.eagle3.whisper import WhisperEagle3DraftModel


def align_whisper_eagle3(row):
    """Pair (auxiliary h[t-1], token x[t]) with logits predicting x[t+1].

    The target is Whisper's pre-LayerNorm h[t]. The final token has no next-token
    label; EOS is supervised through the preceding hidden state, never through
    padding. Rows must be independent utterances before shifting.
    """
    length = row["input_ids"].shape[1]
    if length < 3:  # noqa: PLR2004 -- shifted input plus a real next-token target
        raise ValueError("EAGLE-3 requires at least three decoder tokens")
    labels = torch.cat(
        [row["input_ids"][:, 2:], row["input_ids"].new_zeros(1, 1)], dim=1
    )
    mask = torch.cat([row["loss_mask"][:, 2:], row["loss_mask"].new_zeros(1, 1)], dim=1)
    return {
        "input_ids": row["input_ids"][:, 1:],
        "hidden_states": row["hidden_states"][:, :-1],
        "verifier_last_hidden_states": row["verifier_last_hidden_states"][:, 1:],
        "position_ids": row["position_ids"][:, 1:],
        "document_ids": torch.zeros_like(row["input_ids"][:, 1:]),
        "loss_mask": mask,
    }, labels


def _documents(features):
    """Do not let rollout shifts cross a packed utterance boundary."""
    for document in torch.unique(features["document_ids"]).tolist():
        if document < 0:
            continue
        selected = features["document_ids"][0] == document
        yield {key: value[:, selected] for key, value in features.items()}


def _loss_config(options, labels):
    losses = resolve_loss_config("kl_div", implementation=options.implementation)
    if options.response_ce_weight:

        def response_ce(scores, _targets, labels=labels):
            offset = labels.shape[1] - scores.shape[1]
            return torch.nn.functional.cross_entropy(
                scores.float().reshape(-1, scores.shape[-1]),
                labels[:, offset:].reshape(-1),
                reduction="none",
            ).reshape(scores.shape[:2])

        losses["response_ce"] = (response_ce, options.response_ce_weight)

    return losses


def _policy_processor(options, suppressed, mask, labels):
    audit = {}

    def process_logits(scores, audit=audit, mask=mask, labels=labels):
        # Native EAGLE projects the verifier once before projecting draft steps.
        # Capture the audit at that boundary without a second vocabulary projection.
        if not audit:
            audit["raw_teacher_agreement_sum"] = (
                (scores.argmax(-1) == labels) * mask
            ).sum()
        if options.policy_targets and suppressed:
            scores = scores.index_fill(
                -1,
                torch.tensor(suppressed, device=scores.device),
                torch.finfo(scores.dtype).min,
            )
        if "policy_teacher_agreement_sum" not in audit:
            audit["policy_teacher_agreement_sum"] = (
                (scores.argmax(-1) == labels) * mask
            ).sum()
        return scores

    return process_logits, audit


def eagle3_whisper_loss(
    draft: WhisperEagle3DraftModel, features, *, options, suppressed_tokens=()
):
    """Use native teacher-forced token/unrolled hidden-state training per document.

    Teacher extraction remains batched. Independent native forwards avoid shifted
    tokens leaking across documents during later rollout steps. Loss is a
    response-token-weighted mean of the native depth-decayed rollout objective.
    """
    if options.position_weight == "dpace":
        raise ValueError(
            "Dpace is a DFlash block objective; EAGLE-3 uses rollout decay"
        )
    suppressed = [
        token for token in suppressed_tokens if 0 <= token < draft.lm_head.out_features
    ]
    totals = defaultdict(lambda: torch.zeros((), device=features["input_ids"].device))
    numerator = torch.zeros((), device=features["input_ids"].device)
    denominator = numerator.clone()
    for row in _documents(features):
        batch, labels = align_whisper_eagle3(row)
        mask = batch["loss_mask"]
        count = mask.sum()
        depth = min(draft.block_size - 1, batch["input_ids"].shape[1])
        losses = _loss_config(options, labels)
        process_logits, audit = _policy_processor(options, suppressed, mask, labels)
        _, loss, metrics = draft(
            **batch,
            ttt_steps=depth,
            ttt_step_loss_decay=options.rollout_decay,
            loss_config=losses,
            logits_transform=process_logits,
        )
        numerator = numerator + loss * count
        denominator = denominator + count
        for key, value in metrics.items():
            totals[key] += value.detach()
        for key, value in audit.items():
            totals[key] += value.detach()
        totals["raw_teacher_agreement_total"] += count
        totals["policy_teacher_agreement_total"] += count
        totals["eal_sum"] += count + sum(
            metrics[f"full_acc_{step}_sum"] for step in range(depth)
        )
        totals["eal_total"] += count
        for step in range(depth):
            totals[f"position_{step + 1}_acc_sum"] += metrics[f"full_acc_{step}_sum"]
            totals[f"position_{step + 1}_acc_total"] += metrics[
                f"full_acc_{step}_total"
            ]
    loss = numerator / denominator.clamp_min(1)
    totals["weighted_loss_sum"] = numerator.detach()
    totals["weighted_loss_total"] = denominator
    return loss, dict(totals)
