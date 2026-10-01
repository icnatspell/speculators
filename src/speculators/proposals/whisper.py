"""Greedy Whisper verification for DFlash and EAGLE-3, short audio, batch size one.

The audio encoder and cross-attention cache are reused. Rejected decoder tokens
are removed from the self-attention cache and auxiliary-feature history.
"""

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from transformers import LogitsProcessorList, WhisperForConditionalGeneration
from transformers.generation.logits_process import (
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
)

from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.eagle3.whisper import WhisperEagle3DraftModel
from speculators.proposals.whisper_eagle3 import Eagle3WhisperProposal
from speculators.proposals.whisper_profile import (
    WhisperGenerationTimer,
    WhisperStageTimer,
)


@dataclass
class WhisperDecodeResult:
    tokens: torch.Tensor
    verifier_calls: int
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    stage_seconds: dict[str, float] | None = None
    generation_seconds: float | None = None
    draft_rounds: int = 0
    proposed_by_position: list[int] | None = None
    accepted_by_position: list[int] | None = None
    eligible_by_position: list[int] | None = None


class DFlashWhisperContextCache:
    """Cache projections of verified features; query-block tokens never persist."""

    def __init__(self):
        self.length = 0
        self.layers = []

    def update(self, draft, context):
        if context.shape[1] < self.length:
            raise ValueError("Context cache must receive a growing verified prefix")
        if context.shape[1] == self.length:
            return
        suffix = context[:, self.length :]
        target = draft.hidden_norm(draft.fc(suffix))
        positions = torch.arange(self.length, context.shape[1], device=context.device)[
            None
        ]
        rotary = draft.rotary_emb(target, positions)
        for index, layer in enumerate(draft.layers):
            key, value = layer.self_attn.project_context(target, rotary)
            if self.length:
                old_key, old_value = self.layers[index]
                self.layers[index] = (
                    torch.cat([old_key, key], dim=2),
                    torch.cat([old_value, value], dim=2),
                )
            else:
                self.layers.append((key, value))
        self.length = context.shape[1]


def whisper_processors(teacher, prompt_length: int) -> LogitsProcessorList:
    """Mirror short-form, no-timestamps Whisper token suppression rules.

    The caller supplies the complete language/task/no-timestamps prompt. Beam
    search, sampling, timestamps and long-form fallback are outside this path.
    """
    config = teacher.generation_config
    processors = LogitsProcessorList()
    if config.suppress_tokens:
        processors.append(SuppressTokensLogitsProcessor(config.suppress_tokens))
    if config.begin_suppress_tokens:
        processors.append(
            SuppressTokensAtBeginLogitsProcessor(
                config.begin_suppress_tokens, begin_index=prompt_length
            )
        )
    return processors


def _next_token(processors, prefix, logits):
    return processors(prefix, logits.clone()).argmax(-1, keepdim=True)


class WhisperTokenSelector:
    """Prepare sparse suppression once; preserve arbitrary processors as fallback."""

    def __init__(self, processors, device, enabled=True):
        self.processors = processors
        self.parallel = enabled and all(
            type(p)
            in (SuppressTokensLogitsProcessor, SuppressTokensAtBeginLogitsProcessor)
            for p in processors
        )
        self.suppression = []
        self.prepared_vocab = None
        if self.parallel:
            for processor in processors:
                if type(processor) is SuppressTokensLogitsProcessor:
                    self.suppression.append(
                        (None, processor.suppress_tokens.to(device))
                    )
                else:
                    self.suppression.append(
                        (
                            processor.begin_index,
                            processor.begin_suppress_tokens.to(device),
                        )
                    )

    def block(self, prefix_length, logits):
        if self.prepared_vocab != logits.shape[-1]:
            self.suppression = [
                (begin, indices[(indices >= 0) & (indices < logits.shape[-1])])
                for begin, indices in self.suppression
            ]
            self.prepared_vocab = logits.shape[-1]
        scores = logits.clone()
        for begin, indices in self.suppression:
            if begin is None:
                scores[..., indices] = -float("inf")
            elif prefix_length <= begin < prefix_length + scores.shape[1]:
                scores[:, begin - prefix_length, indices] = -float("inf")
        return scores.argmax(-1)

    def one(self, prefix, logits):
        if self.parallel:
            return self.block(prefix.shape[1], logits[:, None])
        return _next_token(self.processors, prefix, logits)


def _draft_candidates(selector, prefix, anchor, logits, budget, *, eos):
    count = min(logits.shape[1], budget - 1)
    if selector.parallel:
        tokens = selector.block(prefix.shape[1] + 1, logits[:, :count])
        # One host transfer for the block, rather than one synchronization per token.
        values = tokens[0].tolist()
        if eos in values:
            tokens = tokens[:, : values.index(eos) + 1]
        return torch.cat([anchor, tokens], dim=1)
    candidates = anchor
    for index in range(count):
        token = _next_token(
            selector.processors,
            torch.cat([prefix, candidates], dim=1),
            logits[:, index],
        )
        candidates = torch.cat([candidates, token], dim=1)
        if token.item() == eos:
            break
    return candidates


def _accepted_prefix(selector, prefix, candidates, logits, eos):
    if selector.parallel:
        expected = selector.block(
            prefix.shape[1] + 1, logits[:, : candidates.shape[1] - 1]
        )
        matches = expected.eq(candidates[:, 1:])
        # Share a single transfer for acceptance and the accepted EOS decision.
        status = torch.stack([matches[0], candidates[0, 1:].eq(eos)]).tolist()
        accepted = 1
        for match in status[0]:
            if not match:
                break
            accepted += 1
        return accepted, accepted > 1 and status[1][accepted - 2]
    accepted = 1
    for index in range(1, candidates.shape[1]):
        expected = _next_token(
            selector.processors,
            torch.cat([prefix, candidates[:, :index]], dim=1),
            logits[:, index - 1],
        )
        if expected.item() != candidates[:, index].item():
            break
        accepted += 1
    return accepted, candidates[0, accepted - 1].item() == eos


def _auxiliary(output, layer_ids):
    return torch.cat([output.decoder_hidden_states[i] for i in layer_ids], dim=-1)


@torch.no_grad()
def dflash_whisper_proposal(draft, context, anchor, *, context_cache=None):
    """Run one bidirectional query block conditioned on verified features."""
    device = draft.embed_tokens.weight.device
    context = context.to(device)
    anchor = anchor.to(device)
    length = context.shape[1]
    query_ids = torch.full(
        (1, draft.block_size),
        draft.mask_token_id,
        device=context.device,
        dtype=torch.long,
    )
    query_ids[:, :1] = anchor
    query = draft.embed_tokens(query_ids)
    if context_cache is None:
        target = draft.hidden_norm(draft.fc(context))
        positions = torch.arange(length + draft.block_size, device=context.device)[None]
    else:
        context_cache.update(draft, context)
        target = context[:, :0, : draft.hidden_size]
        positions = torch.arange(
            length, length + draft.block_size, device=context.device
        )[None]
    position_embeddings = draft.rotary_emb(context, positions)
    for index, layer in enumerate(draft.layers):
        query = layer(
            hidden_states=query,
            target_hidden=target,
            attention_mask=None,
            position_ids=positions,
            position_embeddings=position_embeddings,
            use_cache=False,
            context_key_values=context_cache.layers[index] if context_cache else None,
        )
    return draft.lm_head(draft.norm(query))[:, 1:]


def _validate_inputs(teacher, prompt, max_new_tokens):
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:  # noqa: PLR2004
        raise ValueError("Expected a nonempty prompt with batch size one")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if prompt.shape[1] + max_new_tokens > teacher.config.max_target_positions:
        raise ValueError("Decode budget exceeds Whisper's decoder position limit")


@torch.no_grad()
def greedy_whisper_decode(
    teacher: WhisperForConditionalGeneration,
    audio: torch.Tensor,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    processors: LogitsProcessorList | None = None,
    profile: bool = False,
    encoder_outputs=None,
    measure_generation: bool = False,
    optimize_decode: bool = True,
) -> WhisperDecodeResult:
    """Cached ordinary greedy decoding, including one audio encoding."""
    _validate_inputs(teacher, prompt, max_new_tokens)
    teacher.eval()
    if processors is None:
        processors = whisper_processors(teacher, prompt.shape[1])
    selector = WhisperTokenSelector(processors, teacher.device, optimize_decode)
    timer = WhisperStageTimer(teacher.device, profile)
    generation_timer = WhisperGenerationTimer(teacher.device, measure_generation)
    with timer.measure("encoder"):
        encoder = encoder_outputs
        if encoder is None:
            encoder = teacher.model.encoder(audio, return_dict=True)
    prefix = prompt.clone()
    with timer.measure("prefill"):
        output = teacher(
            encoder_outputs=encoder, decoder_input_ids=prefix, use_cache=True
        )
    calls = 1
    for _ in range(max_new_tokens):
        with timer.measure("selection"):
            token = selector.one(prefix, output.logits[:, -1])
        prefix = torch.cat([prefix, token], dim=1)
        if token.item() == teacher.config.eos_token_id:
            break
        if prefix.shape[1] == prompt.shape[1] + max_new_tokens:
            break
        generation_timer.begin()
        with timer.measure("verifier"):
            output = teacher(
                encoder_outputs=encoder,
                decoder_input_ids=token,
                past_key_values=output.past_key_values,
                use_cache=True,
            )
        calls += 1
    generation_seconds = generation_timer.finish()
    return WhisperDecodeResult(
        prefix,
        calls,
        stage_seconds=timer.finish(),
        generation_seconds=generation_seconds,
    )


@torch.no_grad()
def speculative_whisper_decode(  # noqa: C901
    teacher: WhisperForConditionalGeneration,
    draft: DFlashDraftModel | WhisperEagle3DraftModel,
    audio: torch.Tensor,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    processors: LogitsProcessorList | None = None,
    proposal_fn: Callable = dflash_whisper_proposal,
    cache_draft_context: bool = False,
    profile: bool = False,
    encoder_outputs=None,
    measure_generation: bool = False,
    optimize_decode: bool = True,
    draft_dtype: torch.dtype | None = None,
    candidate_proposer: Callable | None = None,
) -> WhisperDecodeResult:
    """Verify parallel proposals against causal Whisper greedy predictions.

    Each round begins with a guaranteed teacher token (the anchor), then
    accepts only the contiguous matching draft prefix. The next round starts
    with the correction token on rejection, or a bonus token on full acceptance.
    """
    _validate_inputs(teacher, prompt, max_new_tokens)
    teacher.eval()
    draft.eval()
    if candidate_proposer is None and isinstance(draft, WhisperEagle3DraftModel):
        candidate_proposer = Eagle3WhisperProposal(
            draft, cache_context=cache_draft_context
        )

    def draft_autocast():
        if draft_dtype is None or draft_dtype == torch.float32:
            return nullcontext()
        return torch.autocast(draft.embed_tokens.weight.device.type, dtype=draft_dtype)

    if processors is None:
        processors = whisper_processors(teacher, prompt.shape[1])
    selector = WhisperTokenSelector(processors, teacher.device, optimize_decode)
    timer = WhisperStageTimer(teacher.device, profile)
    generation_timer = WhisperGenerationTimer(
        teacher.device,
        measure_generation,
        extra_devices=[draft.embed_tokens.weight.device],
    )
    with timer.measure("encoder"):
        encoder = encoder_outputs
        if encoder is None:
            encoder = teacher.model.encoder(audio, return_dict=True)
    prefix = prompt.clone()
    with timer.measure("prefill"):
        output = teacher(
            encoder_outputs=encoder,
            decoder_input_ids=prefix,
            output_hidden_states=True,
            use_cache=True,
        )
    context = _auxiliary(output, draft.target_layer_ids)
    cache = output.past_key_values
    next_logits = output.logits[:, -1]
    result = WhisperDecodeResult(
        prefix,
        1,
        proposed_by_position=[0] * (draft.block_size - 1),
        accepted_by_position=[0] * (draft.block_size - 1),
        eligible_by_position=[0] * (draft.block_size - 1),
    )
    limit = prompt.shape[1] + max_new_tokens
    draft_cache = (
        DFlashWhisperContextCache()
        if cache_draft_context and candidate_proposer is None
        else None
    )
    if draft_cache is not None and proposal_fn is dflash_whisper_proposal:
        with timer.measure("draft_prefill"), draft_autocast():
            draft_cache.update(draft, context.to(draft.embed_tokens.weight.device))
    if isinstance(candidate_proposer, Eagle3WhisperProposal):
        with timer.measure("draft_prefill"), draft_autocast():
            candidate_proposer.prefill(context, prefix)
    while prefix.shape[1] < limit:
        with timer.measure("selection"):
            anchor = selector.one(prefix, next_logits)
        if anchor.item() == teacher.config.eos_token_id:
            prefix = torch.cat([prefix, anchor], dim=1)
            break
        if prefix.shape[1] + 1 == limit:
            prefix = torch.cat([prefix, anchor], dim=1)
            break
        generation_timer.begin()
        with timer.measure("draft"), draft_autocast():
            if candidate_proposer is not None:
                candidates = candidate_proposer(
                    context,
                    prefix,
                    anchor,
                    budget=limit - prefix.shape[1],
                    eos=teacher.config.eos_token_id,
                    select_token=selector.one,
                )
            elif proposal_fn is dflash_whisper_proposal:
                proposal_logits = proposal_fn(
                    draft,
                    context.to(draft.embed_tokens.weight.device),
                    anchor.to(draft.embed_tokens.weight.device),
                    context_cache=draft_cache,
                )
            else:
                proposal_logits = proposal_fn(draft, context, anchor)
            if candidate_proposer is None:
                proposal_logits = proposal_logits.to(teacher.device)
        with timer.measure("candidate_selection"):
            if candidate_proposer is None:
                candidates = _draft_candidates(
                    selector,
                    prefix,
                    anchor,
                    proposal_logits,
                    limit - prefix.shape[1],
                    eos=teacher.config.eos_token_id,
                )
        result.draft_rounds += 1
        result.proposed_tokens += candidates.shape[1] - 1
        for position in range(candidates.shape[1] - 1):
            result.proposed_by_position[position] += 1
        with timer.measure("verifier"):
            verified = teacher(
                encoder_outputs=encoder,
                decoder_input_ids=candidates,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
        result.verifier_calls += 1
        with timer.measure("acceptance"):
            accepted, finished = _accepted_prefix(
                selector,
                prefix,
                candidates,
                verified.logits,
                teacher.config.eos_token_id,
            )
        result.accepted_tokens += accepted - 1
        for position in range(candidates.shape[1] - 1):
            if position == 0 or accepted - 1 >= position:
                result.eligible_by_position[position] += 1
        for position in range(accepted - 1):
            result.accepted_by_position[position] += 1
        rejected = candidates.shape[1] - accepted
        result.rejected_tokens += rejected
        with timer.measure("cache_history"):
            cache = verified.past_key_values
            if rejected:
                cache.crop(-rejected)
            prefix = torch.cat([prefix, candidates[:, :accepted]], dim=1)
            context = torch.cat(
                [context, _auxiliary(verified, draft.target_layer_ids)[:, :accepted]],
                dim=1,
            )
            if cache.get_seq_length() != prefix.shape[1]:
                raise RuntimeError("Whisper cache did not match the accepted prefix")
            next_logits = verified.logits[:, accepted - 1]
        if finished:
            break
    result.tokens = prefix
    result.generation_seconds = generation_timer.finish()
    result.stage_seconds = timer.finish()
    return result
