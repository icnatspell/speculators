"""Whisper adaptation gates around native EAGLE-3 training and verification."""

import copy

import pytest
import torch

from speculators.data_generation.whisper import WhisperFeatureAdapter
from speculators.models.eagle3 import Eagle3DraftModel
from speculators.models.eagle3.whisper import WhisperEagle3DraftModel
from speculators.proposals.whisper import (
    WhisperTokenSelector,
    greedy_whisper_decode,
    speculative_whisper_decode,
    whisper_processors,
)
from speculators.proposals.whisper_eagle3 import Eagle3WhisperProposal
from speculators.train.whisper import (
    WhisperLossOptions,
    build_whisper_draft,
    load_whisper_draft,
    save_whisper_draft,
    whisper_draft_loss,
)
from speculators.train.whisper_eagle3 import align_whisper_eagle3
from speculators.train.whisper_runtime import pack_whisper_features
from tests.unit.test_whisper_features import tiny_teacher


def make_models():
    torch.manual_seed(31)
    teacher = tiny_teacher().eval().requires_grad_(False)
    draft = build_whisper_draft(teacher, [0, 1, 2], algorithm="eagle3")
    return teacher, draft


def feature_row(teacher, ids=(1, 3, 4, 5, 2)):
    return WhisperFeatureAdapter(teacher, [0, 1, 2]).extract(
        torch.randn(1, 4, 16), torch.tensor([ids]), prompt_length=2
    )


def test_shifted_targets_include_eos_but_exclude_terminal_padding():
    teacher, _ = make_models()
    row = feature_row(teacher)
    aligned, labels = align_whisper_eagle3(row)
    assert aligned["input_ids"].tolist() == [[3, 4, 5, 2]]
    assert labels.tolist() == [[4, 5, 2, 0]]
    assert aligned["loss_mask"].tolist() == [[1, 1, 1, 0]]
    assert aligned["position_ids"].tolist() == [[1, 2, 3, 4]]
    torch.testing.assert_close(aligned["hidden_states"], row["hidden_states"][:, :-1])
    torch.testing.assert_close(
        aligned["verifier_last_hidden_states"],
        row["verifier_last_hidden_states"][:, 1:],
    )


def test_whisper_norm_and_full_native_checkpoint_roundtrip(tmp_path):
    teacher, draft = make_models()
    assert isinstance(draft, Eagle3DraftModel)
    assert not draft.lm_head.weight.requires_grad  # native verifier-owned head
    assert not draft.embed_tokens.weight.requires_grad
    assert not draft.verifier_lm_head.weight.requires_grad
    assert not draft.verifier_norm.bias.requires_grad
    raw = torch.randn(1, 3, teacher.config.d_model)
    with torch.no_grad():
        torch.testing.assert_close(
            draft.verifier_lm_head(draft.verifier_norm(raw)),
            teacher.proj_out(teacher.model.decoder.layer_norm(raw)),
        )
    save_whisper_draft(draft, tmp_path)
    restored = load_whisper_draft(tmp_path)
    assert isinstance(restored, WhisperEagle3DraftModel)
    assert restored.block_size == 4
    assert not restored.lm_head.weight.requires_grad
    assert not restored.verifier_norm.bias.requires_grad
    for name, tensor in draft.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)


def test_native_rollouts_update_draft_and_leave_teacher_frozen(monkeypatch):
    teacher, draft = make_models()
    row = feature_row(teacher, (1, 3, 4, 5, 6, 2))
    before = {name: value.clone() for name, value in teacher.state_dict().items()}
    fc_before = draft.fc.weight.detach().clone()
    projections = []
    original = draft.verifier_lm_head.forward

    def project(*args, **kwargs):
        projections.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(draft.verifier_lm_head, "forward", project)
    optimizer = torch.optim.AdamW(
        [p for p in draft.parameters() if p.requires_grad], lr=0.01
    )
    with torch.compiler.set_stance("force_eager"):
        loss, metrics = whisper_draft_loss(
            draft,
            row,
            options=WhisperLossOptions(response_ce_weight=0.1),
            suppressed_tokens=[10, 11, 99999],
        )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in draft.parameters() if p.grad is not None
    )
    optimizer.step()
    assert projections == [True]  # policy audit reuses the native target projection
    assert not torch.equal(fc_before, draft.fc.weight)
    assert all(
        torch.equal(value, teacher.state_dict()[name]) for name, value in before.items()
    )
    assert metrics["weighted_loss_total"] == 4
    assert all(f"loss_{step}_sum" in metrics for step in range(3))
    assert 1 <= metrics["eal_sum"] / metrics["eal_total"] <= 4


def test_packed_utterances_match_independent_native_losses_and_gradients():
    teacher, draft = make_models()
    reference = copy.deepcopy(draft)
    rows = [feature_row(teacher), feature_row(teacher, (1, 3, 6, 7, 8, 9, 2))]
    with torch.compiler.set_stance("force_eager"):
        packed_loss, packed_metrics = whisper_draft_loss(
            draft, pack_whisper_features(rows)
        )
        parts = [whisper_draft_loss(reference, row) for row in rows]
    weights = [metrics["weighted_loss_total"] for _, metrics in parts]
    expected = sum(
        loss * weight for (loss, _), weight in zip(parts, weights, strict=True)
    ) / sum(weights)
    torch.testing.assert_close(packed_loss, expected)
    packed_loss.backward()
    expected.backward()
    for actual, individual in zip(
        draft.parameters(), reference.parameters(), strict=True
    ):
        if actual.grad is not None:
            torch.testing.assert_close(
                actual.grad, individual.grad, rtol=1e-4, atol=1e-6
            )
    assert packed_metrics["weighted_loss_total"] == sum(weights)


@pytest.mark.parametrize("cache_context", [False, True])
@pytest.mark.parametrize("budget", [1, 2, 7, 12])
def test_native_eagle_proposals_preserve_teacher_tokens(cache_context, budget):
    teacher, draft = make_models()
    teacher.generation_config.suppress_tokens = [teacher.config.eos_token_id]
    audio, prompt = torch.randn(1, 4, 16), torch.tensor([[1, 3]])
    baseline = greedy_whisper_decode(teacher, audio, prompt, max_new_tokens=budget)
    speculative = speculative_whisper_decode(
        teacher,
        draft,
        audio,
        prompt,
        max_new_tokens=budget,
        cache_draft_context=cache_context,
    )
    assert torch.equal(baseline.tokens, speculative.tokens)
    assert len(speculative.proposed_by_position) == 3


def test_cached_proposals_match_rebuild_and_discard_hypothetical_suffixes():
    teacher, draft = make_models()
    teacher.generation_config.suppress_tokens = [teacher.config.eos_token_id]
    audio, prompt = torch.randn(1, 4, 16), torch.tensor([[1, 3]])
    gold = greedy_whisper_decode(teacher, audio, prompt, max_new_tokens=10).tokens
    context = WhisperFeatureAdapter(teacher, draft.target_layer_ids).extract(
        audio, gold, prompt_length=2
    )["hidden_states"]
    selector = WhisperTokenSelector(
        whisper_processors(teacher, 2), torch.device("cpu"), True
    )
    cached = Eagle3WhisperProposal(draft, cache_context=True)
    for length in [2, 4, 7, 3]:  # includes shortening/rollback of an established cache
        kwargs = {
            "budget": 4,
            "eos": teacher.config.eos_token_id,
            "select_token": selector.one,
        }
        actual = cached(
            context[:, :length],
            gold[:, :length],
            gold[:, length : length + 1],
            **kwargs,
        )
        expected = Eagle3WhisperProposal(draft, cache_context=False)(
            context[:, :length],
            gold[:, :length],
            gold[:, length : length + 1],
            **kwargs,
        )
        assert torch.equal(actual, expected)
        assert cached.cache.get_seq_length() == length - 1


def test_proposer_stops_at_eos_and_respects_remaining_budget():
    teacher, draft = make_models()
    row = feature_row(teacher)
    proposer = Eagle3WhisperProposal(draft)
    prefix, anchor = row["input_ids"][:, :2], row["input_ids"][:, 2:3]
    candidates = proposer(
        row["hidden_states"][:, :2],
        prefix,
        anchor,
        budget=4,
        eos=2,
        select_token=lambda _prefix, _logits: torch.tensor([[2]]),
    )
    assert candidates.tolist() == [[4, 2]]
    assert proposer.cache.get_seq_length() == 1
    candidates = proposer(
        row["hidden_states"][:, :2],
        prefix,
        anchor,
        budget=2,
        eos=2,
        select_token=lambda _prefix, _logits: torch.tensor([[6]]),
    )
    assert candidates.tolist() == [[4, 6]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused KL regression")
def test_cuda_native_eagle_fused_sdpa_matches_eager():
    teacher, reference = make_models()
    teacher = teacher.cuda()
    reference = reference.cuda()
    optimized = build_whisper_draft(
        teacher, [0, 1, 2], algorithm="eagle3", attention_implementation="sdpa"
    )
    optimized.load_state_dict(reference.state_dict())
    row = WhisperFeatureAdapter(teacher, [0, 1, 2]).extract(
        torch.randn(1, 4, 16, device="cuda"),
        torch.tensor([[1, 3, 4, 5, 6, 2]], device="cuda"),
        prompt_length=2,
    )
    losses = []
    with torch.compiler.set_stance("force_eager"):
        for draft, implementation in [(reference, "eager"), (optimized, "fused")]:
            loss, _ = whisper_draft_loss(
                draft,
                row,
                options=WhisperLossOptions(implementation=implementation),
                suppressed_tokens=[10, 11],
            )
            loss.backward()
            losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-4, atol=1e-5)
    for expected, actual in zip(
        reference.parameters(), optimized.parameters(), strict=True
    ):
        if expected.grad is not None:
            torch.testing.assert_close(expected.grad, actual.grad, rtol=1e-3, atol=1e-5)
