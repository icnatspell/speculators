import pytest
import torch
from transformers import (
    LogitsProcessorList,
    WhisperConfig,
    WhisperForConditionalGeneration,
)

from speculators.data_generation.whisper import WhisperFeatureAdapter
from speculators.proposals.whisper import (
    DFlashWhisperContextCache,
    WhisperTokenSelector,
    _accepted_prefix,
    _draft_candidates,
    dflash_whisper_proposal,
    greedy_whisper_decode,
    speculative_whisper_decode,
    whisper_processors,
)
from speculators.proposals.whisper_profile import WhisperGenerationTimer
from speculators.train.whisper import (
    build_whisper_draft,
    generate_whisper_tokens,
    load_whisper_draft,
    save_whisper_draft,
    train_whisper_step,
)


def tiny_teacher():
    return WhisperForConditionalGeneration(
        WhisperConfig(
            vocab_size=32,
            num_mel_bins=4,
            d_model=16,
            encoder_layers=1,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_source_positions=8,
            max_target_positions=16,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
        )
    )


def test_features_reconstruct_teacher_logits_and_mask_next_tokens():
    torch.manual_seed(0)
    teacher = tiny_teacher()
    with torch.no_grad():
        teacher.model.decoder.layer_norm.weight.fill_(1.7)
        teacher.model.decoder.layer_norm.bias.fill_(0.3)
    adapter = WhisperFeatureAdapter(teacher, [0, 1, 2])
    audio = torch.randn(1, 4, 16)
    tokens = torch.tensor([[1, 3, 4, 5, 6, 2]])
    features = adapter.extract(audio, tokens, prompt_length=2)
    with torch.no_grad():
        expected = teacher(
            input_features=audio, decoder_input_ids=tokens, use_cache=False
        ).logits
        actual = teacher.proj_out(
            teacher.model.decoder.layer_norm(features["verifier_last_hidden_states"])
        )
    torch.testing.assert_close(actual, expected)
    assert features["hidden_states"].shape == (1, 6, 48)
    assert features["loss_mask"].tolist() == [[0, 0, 1, 1, 1, 1]]
    assert not any(value.requires_grad for value in features.values())
    assert not any(param.requires_grad for param in teacher.parameters())
    assert not teacher.model.decoder.layer_norm._forward_pre_hooks


def test_teacher_generation_preserves_prompt_and_eos():
    teacher = tiny_teacher().eval()
    teacher.generation_config.begin_suppress_tokens = None
    teacher.generation_config.suppress_tokens = None
    with torch.no_grad():
        teacher.proj_out.weight.zero_()
    prompt = torch.tensor([[1, 3]])

    # A generation-config forced token makes the raw result terminate at EOS.
    teacher.generation_config.forced_eos_token_id = teacher.config.eos_token_id
    tokens = generate_whisper_tokens(
        teacher, torch.randn(1, 4, 16), prompt, max_new_tokens=2
    )
    assert tokens[:, :2].tolist() == prompt.tolist()
    assert tokens[0, -1].item() == teacher.config.eos_token_id


@pytest.mark.parametrize("layers", [[], [0, 0], [-1], [3]])
def test_invalid_layers(layers):
    with pytest.raises(ValueError, match="Layer IDs"):
        WhisperFeatureAdapter(tiny_teacher(), layers)


def test_draft_update_targets_and_checkpoint_reload(tmp_path):
    torch.manual_seed(7)
    teacher = tiny_teacher()
    with torch.no_grad():
        teacher.model.decoder.layer_norm.weight.fill_(1.7)
        teacher.model.decoder.layer_norm.bias.fill_(0.3)
    adapter = WhisperFeatureAdapter(teacher, [0, 1])
    draft = build_whisper_draft(teacher, adapter.target_layer_ids)
    features = adapter.extract(
        torch.randn(1, 4, 16),
        torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9, 10, 2]]),
        prompt_length=2,
    )
    with torch.no_grad():
        _, _, targets, mask, indices = draft._backbone_forward(
            **features, max_anchors=2
        )
        logits = teacher.proj_out(
            teacher.model.decoder.layer_norm(features["verifier_last_hidden_states"])
        )
        torch.testing.assert_close(targets, logits[:, (indices - 1) % logits.shape[1]])
        assert mask.sum() > 0
    before = draft.fc.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [p for p in draft.parameters() if p.requires_grad], lr=1e-3
    )
    loss = train_whisper_step(draft, optimizer, features, max_anchors=2)
    assert loss >= 0
    assert not torch.equal(before, draft.fc.weight)
    assert all(p.grad is None for p in teacher.parameters())
    save_whisper_draft(draft, tmp_path)
    restored = load_whisper_draft(tmp_path)
    assert restored._attn_impl == "eager"
    for key, value in draft.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    assert not any(p.requires_grad for p in restored.verifier_norm.parameters())
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "draft.safetensors",
        "whisper_draft.json",
    ]


@pytest.mark.parametrize("budget", [1, 2, 7, 12])
def test_speculative_decode_matches_greedy_with_real_draft(budget):
    torch.manual_seed(42)
    teacher = tiny_teacher().eval()
    draft = build_whisper_draft(teacher, [0, 1])
    audio = torch.randn(1, 4, 16)
    prompt = torch.tensor([[1, 3]])
    baseline = greedy_whisper_decode(
        teacher, audio, prompt, max_new_tokens=budget, processors=LogitsProcessorList()
    )
    actual = speculative_whisper_decode(
        teacher,
        draft,
        audio,
        prompt,
        max_new_tokens=budget,
        processors=LogitsProcessorList(),
    )
    assert torch.equal(actual.tokens, baseline.tokens)


@pytest.mark.parametrize("proposal_token", [0, 1])
def test_forced_acceptance_and_rejection(proposal_token):
    teacher = tiny_teacher().eval()
    with torch.no_grad():
        teacher.proj_out.weight.zero_()
    draft = build_whisper_draft(teacher, [0, 1])

    def fixed_proposal(_draft, context, _anchor):
        logits = context.new_full((1, 3, 32), -100)
        logits[:, :, proposal_token] = 100
        return logits

    audio = torch.randn(1, 4, 16)
    prompt = torch.tensor([[1, 3]])
    actual = speculative_whisper_decode(
        teacher,
        draft,
        audio,
        prompt,
        max_new_tokens=8,
        processors=LogitsProcessorList(),
        proposal_fn=fixed_proposal,
    )
    assert actual.tokens.tolist() == [[1, 3] + [0] * 8]
    if proposal_token == 0:
        assert actual.accepted_tokens == 6
        assert actual.rejected_tokens == 0
        assert actual.verifier_calls == 3
    else:
        assert actual.accepted_tokens == 0
        assert actual.rejected_tokens > 0
        assert actual.verifier_calls == 8


def test_eos_inside_draft_and_first_token_suppression():
    teacher = tiny_teacher().eval()
    draft = build_whisper_draft(teacher, [0, 1])

    def choose_by_length(input_ids, scores):
        scores.fill_(-100)
        scores[:, 2 if input_ids.shape[1] >= 4 else 0] = 100
        return scores

    processors = LogitsProcessorList([choose_by_length])
    audio = torch.randn(1, 4, 16)
    prompt = torch.tensor([[1, 3]])
    actual = speculative_whisper_decode(
        teacher, draft, audio, prompt, max_new_tokens=10, processors=processors
    )
    baseline = greedy_whisper_decode(
        teacher, audio, prompt, max_new_tokens=10, processors=processors
    )
    assert actual.tokens.tolist() == [[1, 3, 0, 0, 2]]
    assert torch.equal(actual.tokens, baseline.tokens)
    assert actual.accepted_tokens == 2


def test_draft_context_cache_matches_uncached_and_only_projects_suffixes():
    torch.manual_seed(42)
    teacher = tiny_teacher().eval()
    draft = build_whisper_draft(teacher, [0, 1], num_layers=2).eval()
    context = torch.randn(1, 11, 32)
    anchor = torch.tensor([[3]])
    cache = DFlashWhisperContextCache()
    projected_lengths = []

    def count_projection(_module, args):
        projected_lengths.append(args[0].shape[1])

    for length in (2, 5, 11, 11):
        expected = dflash_whisper_proposal(draft, context[:, :length], anchor)
        hook = draft.fc.register_forward_pre_hook(count_projection)
        try:
            actual = dflash_whisper_proposal(
                draft, context[:, :length], anchor, context_cache=cache
            )
        finally:
            hook.remove()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        assert cache.length == length
        assert all(key.shape[2] == length for key, _ in cache.layers)
    assert projected_lengths == [2, 3, 6]
    with pytest.raises(ValueError, match="growing verified prefix"):
        dflash_whisper_proposal(draft, context[:, :5], anchor, context_cache=cache)


def test_cached_and_uncached_decode_match_and_profile_stages():
    torch.manual_seed(42)
    teacher = tiny_teacher().eval()
    draft = build_whisper_draft(teacher, [0, 1]).eval()
    audio = torch.randn(1, 4, 16)
    prompt = torch.tensor([[1, 3]])
    processors = LogitsProcessorList()
    uncached = speculative_whisper_decode(
        teacher,
        draft,
        audio,
        prompt,
        max_new_tokens=12,
        processors=processors,
        cache_draft_context=False,
    )
    cached = speculative_whisper_decode(
        teacher,
        draft,
        audio,
        prompt,
        max_new_tokens=12,
        processors=processors,
        cache_draft_context=True,
        profile=True,
    )
    assert torch.equal(cached.tokens, uncached.tokens)
    assert cached.accepted_tokens == uncached.accepted_tokens
    assert cached.rejected_tokens == uncached.rejected_tokens
    assert set(cached.stage_seconds) == {
        "encoder",
        "prefill",
        "draft_prefill",
        "selection",
        "candidate_selection",
        "acceptance",
        "cache_history",
        "draft",
        "verifier",
        "other",
        "total",
    }
    assert all(value >= 0 for value in cached.stage_seconds.values())
    with torch.no_grad():
        encoder = teacher.model.encoder(audio, return_dict=True)
    without_encoder = greedy_whisper_decode(
        teacher,
        audio,
        prompt,
        max_new_tokens=12,
        processors=processors,
        encoder_outputs=encoder,
    )
    assert torch.equal(cached.tokens, without_encoder.tokens)


def test_generation_timer_excludes_preparation_and_starts_once(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "speculators.proposals.whisper_profile.time.perf_counter", lambda: now[0]
    )
    timer = WhisperGenerationTimer(torch.device("cpu"), True)
    now[0] = 500.0
    timer.begin()
    now[0] = 502.0
    timer.begin()
    now[0] = 503.0
    assert timer.finish() == 3.0
    assert WhisperGenerationTimer(torch.device("cpu"), True).finish() == 0.0
    assert WhisperGenerationTimer(torch.device("cpu"), False).finish() is None


@pytest.mark.parametrize("length", [2, 3, 6])
def test_parallel_token_selection_matches_sequential_suppression(length):
    teacher = tiny_teacher()
    teacher.generation_config.suppress_tokens = [0, 4, 7]
    teacher.generation_config.begin_suppress_tokens = [2, 9]
    processors = whisper_processors(teacher, 3)
    selector = WhisperTokenSelector(processors, torch.device("cpu"))
    logits = torch.randn(1, 4, 32)
    original = logits.clone()
    actual = selector.block(length, logits)
    expected = torch.cat(
        [
            processors(
                torch.ones(1, length + i, dtype=torch.long), logits[:, i].clone()
            ).argmax(-1, keepdim=True)
            for i in range(4)
        ],
        dim=1,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(logits, original)


@pytest.mark.parametrize("budget", [1, 2, 7, 12])
def test_optimized_and_reference_decode_preserve_tokens_and_counts(budget):
    torch.manual_seed(42)
    teacher = tiny_teacher().eval()
    teacher.generation_config.suppress_tokens = [0, 4, 7]
    teacher.generation_config.begin_suppress_tokens = [2, 9]
    draft = build_whisper_draft(teacher, [0, 1])
    audio = torch.randn(1, 4, 16)
    prompt = torch.tensor([[1, 3]])
    reference = speculative_whisper_decode(
        teacher, draft, audio, prompt, max_new_tokens=budget, optimize_decode=False
    )
    actual = speculative_whisper_decode(
        teacher, draft, audio, prompt, max_new_tokens=budget
    )
    assert torch.equal(actual.tokens, reference.tokens)
    assert (
        actual.verifier_calls,
        actual.accepted_tokens,
        actual.proposed_tokens,
        actual.rejected_tokens,
    ) == (
        reference.verifier_calls,
        reference.accepted_tokens,
        reference.proposed_tokens,
        reference.rejected_tokens,
    )


@pytest.mark.parametrize(
    ("predictions", "expected"),
    [
        ([4, 2, 7], (3, True)),
        ([8, 2, 7], (1, False)),
        ([4, 8, 7], (2, False)),
        ([4, 2, 8], (3, True)),
    ],
)
def test_parallel_acceptance_uses_contiguous_matches_and_accepted_eos(
    predictions, expected
):
    selector = WhisperTokenSelector(LogitsProcessorList(), torch.device("cpu"))
    candidates = torch.tensor([[0, 4, 2]])
    logits = torch.full((1, 3, 32), -100.0)
    for index, token in enumerate(predictions):
        logits[0, index, token] = 100
    assert (
        _accepted_prefix(selector, torch.tensor([[1, 3]]), candidates, logits, 2)
        == expected
    )


def test_parallel_candidates_stop_at_first_eos():
    selector = WhisperTokenSelector(LogitsProcessorList(), torch.device("cpu"))
    logits = torch.full((1, 3, 32), -100.0)
    logits[0, 0, 4] = logits[0, 1, 2] = logits[0, 2, 7] = 100
    actual = _draft_candidates(
        selector, torch.tensor([[1, 3]]), torch.tensor([[0]]), logits, 4, eos=2
    )
    assert actual.tolist() == [[0, 4, 2]]


def test_feature_extraction_reuses_encoder_and_preserves_targets():
    teacher = tiny_teacher().eval()
    adapter = WhisperFeatureAdapter(teacher, [0, 1])
    audio = torch.randn(1, 4, 16)
    tokens = torch.tensor([[1, 3, 4, 5, 6, 7, 2]])
    expected = adapter.extract(audio, tokens, prompt_length=2)
    with torch.no_grad():
        encoder = teacher.model.encoder(audio, return_dict=True)
    calls = []
    hook = teacher.model.encoder.register_forward_hook(lambda *_args: calls.append(1))
    try:
        actual = adapter.extract(
            audio, tokens, prompt_length=2, encoder_outputs=encoder
        )
    finally:
        hook.remove()
    assert not calls
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
