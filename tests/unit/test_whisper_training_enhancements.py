"""Regression gates for batched online training, terminal targets and isolation."""

import pytest
import torch

from speculators.data_generation.whisper import WhisperFeatureAdapter
from speculators.proposals.whisper import (
    WhisperDecodeResult,
    speculative_whisper_decode,
)
from speculators.train.whisper import (
    WhisperLossOptions,
    build_whisper_draft,
    whisper_draft_loss,
)
from speculators.train.whisper_eval import select_balanced_ids, summarize_records
from speculators.train.whisper_online import corpus_error_rates, prefetch_map
from speculators.train.whisper_runtime import (
    audio_path,
    batched_rows,
    dataset_identity,
    pack_whisper_features,
    recover_checkpoint,
    repair_jsonl_tail,
)
from tests.unit.test_whisper_features import tiny_teacher


def test_batched_selected_features_match_single_rows_without_lm_head(monkeypatch):
    torch.manual_seed(5)
    teacher = tiny_teacher()
    adapter = WhisperFeatureAdapter(teacher, [0, 1, 2])
    audio = torch.randn(2, 4, 16)
    sequences = [torch.tensor([1, 3, 4, 5, 2]), torch.tensor([1, 3, 6, 7, 8, 9, 2])]
    singles = [
        adapter.extract(audio[i : i + 1], ids[None], prompt_length=2)
        for i, ids in enumerate(sequences)
    ]

    def unused_projection(*args, **kwargs):
        raise AssertionError("Feature extraction must omit vocabulary projection")

    monkeypatch.setattr(teacher.proj_out, "forward", unused_projection)
    batched = adapter.extract_batch(
        audio, sequences, prompt_lengths=[2, 2], pad_to_multiple=8
    )
    for single, batch in zip(singles, batched, strict=True):
        for key in single:
            torch.testing.assert_close(single[key], batch[key], atol=1e-5, rtol=1e-5)


def test_partial_blocks_supervise_eos_and_never_cross_packed_documents():
    torch.manual_seed(5)
    teacher = tiny_teacher()
    adapter = WhisperFeatureAdapter(teacher, [0, 1])
    rows = [
        adapter.extract(torch.randn(1, 4, 16), torch.tensor([ids]), prompt_length=2)
        for ids in [[1, 3, 4, 5, 2], [1, 3, 6, 7, 8, 2]]
    ]
    packed = pack_whisper_features(rows)
    draft = build_whisper_draft(teacher, [0, 1])
    with torch.compiler.set_stance("force_eager"):
        _, logits, _, mask, indices = draft._backbone_forward(
            **packed,
            max_anchors=8,
            allow_partial_blocks=True,
            anchors_per_document=True,
        )
    supervised = indices[mask[0].bool()].tolist()
    assert 4 in supervised
    assert 10 in supervised  # both EOS tokens
    assert not any(index in supervised for index in [5, 6])  # second prompt
    # Editing another utterance cannot change queries anchored in document zero.
    modified = {key: value.clone() for key, value in packed.items()}
    modified["hidden_states"][:, 5:] += 100
    torch.manual_seed(10)
    with torch.compiler.set_stance("force_eager"):
        _, before, _, _, before_indices = draft._backbone_forward(
            **packed,
            max_anchors=8,
            allow_partial_blocks=True,
            anchors_per_document=True,
        )
    torch.manual_seed(10)
    with torch.compiler.set_stance("force_eager"):
        _, after, _, _, after_indices = draft._backbone_forward(
            **modified,
            max_anchors=8,
            allow_partial_blocks=True,
            anchors_per_document=True,
        )
    assert torch.equal(before_indices, after_indices)
    first_doc_queries = before_indices.reshape(-1, 4)[:, 0] < 5
    torch.testing.assert_close(
        before.reshape(8, 4, -1)[first_doc_queries],
        after.reshape(8, 4, -1)[first_doc_queries],
    )
    assert torch.isfinite(logits).all()


def test_policy_suppression_and_response_ce_are_finite_with_gradients():
    teacher = tiny_teacher()
    adapter = WhisperFeatureAdapter(teacher, [0, 1])
    features = adapter.extract(
        torch.randn(1, 4, 16), torch.tensor([[1, 3, 4, 5, 6, 2]]), prompt_length=2
    )
    draft = build_whisper_draft(teacher, [0, 1])
    with torch.compiler.set_stance("force_eager"):
        loss, metrics = whisper_draft_loss(
            draft,
            features,
            max_anchors=4,
            options=WhisperLossOptions(response_ce_weight=0.2),
            suppressed_tokens=[10, 11, 99999],
        )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in draft.parameters() if p.grad is not None
    )
    assert metrics["weighted_loss_total"] > 0
    assert "response_ce_loss_sum" in metrics


def test_producer_pause_quiesces_teacher_hooks():
    teacher = tiny_teacher().eval().requires_grad_(False)
    adapter = WhisperFeatureAdapter(teacher, [0, 1])
    audio = torch.randn(1, 4, 16)
    tokens = torch.tensor([[1, 3, 4, 5, 2]])
    calls = []

    def prepare(index):
        calls.append(index)
        return adapter.extract(audio, tokens, prompt_length=2)

    items = prefetch_map(range(20), prepare, capacity=1)
    next(items)
    with items.pause():
        before = len(calls)
        # Several evaluation forwards cannot enter the extraction hooks.
        for _ in range(3):
            with torch.no_grad():
                teacher(input_features=audio, decoder_input_ids=tokens, use_cache=False)
        assert len(calls) == before
        assert not items.active
    assert next(items)["input_ids"].shape == tokens.shape
    items.close()
    assert not items.worker.is_alive()


def test_balanced_selection_covers_speakers_and_is_repeatable():
    rows = [
        {"id": f"{speaker}-{index}", "speaker_id": speaker, "duration": index * 3 + 1}
        for speaker in range(4)
        for index in range(8)
    ]
    selected = select_balanced_ids(rows, 8, 7)
    assert selected == select_balanced_ids(rows, 8, 7)
    assert {value.split("-")[0] for value in selected} == {"0", "1", "2", "3"}


def test_english_wer_handles_numbers_and_contractions():
    rates = corpus_error_rates(["I can't buy twenty dollars."], ["I can not buy $20."])
    assert rates["wer"] == 0
    assert rates["reference_words"] > 0


def test_data_identity_changes_with_rows_or_preparation(tmp_path):
    (tmp_path / "rows.arrow").write_bytes(b"first")
    first = dataset_identity(tmp_path)
    (tmp_path / "rows.arrow").write_bytes(b"second")
    assert dataset_identity(tmp_path) != first
    second = dataset_identity(tmp_path)
    (tmp_path / "preparation_manifest.json").write_text('{"seq_length":32}')
    assert dataset_identity(tmp_path) != second


def test_interrupted_jsonl_repairs_only_final_partial_row(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_bytes(b'{"id":"one"}\n{"id":')
    repair_jsonl_tail(path)
    assert path.read_bytes() == b'{"id":"one"}\n'
    path.write_bytes(b'bad\n{"id":"two"}\n')
    with pytest.raises(ValueError, match="interior"):
        repair_jsonl_tail(path)


def test_checkpoint_recovers_previous_snapshot(tmp_path):
    backup = tmp_path / "latest.previous"
    backup.mkdir()
    for name in [
        "draft.safetensors",
        "whisper_draft.json",
        "results.json",
        "trainer_state.pt",
    ]:
        (backup / name).write_bytes(b"complete")
    result = recover_checkpoint(tmp_path / "latest")
    assert result == tmp_path / "latest"
    assert result.exists()
    assert not backup.exists()


def test_audio_root_remaps_relative_paths_and_rejects_traversal(tmp_path):
    row = {
        "audio_url": "file:///old/machine/audio/one.flac",
        "audio_relative_path": "one.flac",
    }
    assert audio_path(row, tmp_path) == tmp_path / "one.flac"
    row["audio_relative_path"] = "../outside.flac"
    with pytest.raises(ValueError, match="escapes"):
        audio_path(row, tmp_path)


def test_length_buckets_are_bounded_and_replayable():
    rows = [
        {"id": i, "input_ids": [0] * width} for i, width in enumerate([8, 3, 9, 2, 7])
    ]
    first = list(batched_rows(iter(rows), 2, bucket_buffer=4))
    assert [[row["id"] for row in batch] for batch in first] == [[3, 1], [0, 2], [4]]
    assert first == list(batched_rows(iter(rows), 2, bucket_buffer=4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused KL")
def test_cuda_sdpa_fused_loss_matches_eager_gradients():
    torch.manual_seed(15)
    teacher = tiny_teacher().cuda().eval().requires_grad_(False)
    features = WhisperFeatureAdapter(teacher, [0, 1]).extract(
        torch.randn(1, 4, 16, device="cuda"),
        torch.tensor([[1, 3, 4, 5, 6, 2]], device="cuda"),
        prompt_length=2,
    )
    eager = build_whisper_draft(teacher, [0, 1]).cuda()
    fast = build_whisper_draft(teacher, [0, 1], attention_implementation="sdpa").cuda()
    fast.load_state_dict(eager.state_dict())
    losses = []
    for model, implementation in [(eager, "eager"), (fast, "fused")]:
        torch.manual_seed(25)
        with torch.compiler.set_stance("force_eager"):
            loss, _ = whisper_draft_loss(
                model,
                features,
                max_anchors=4,
                options=WhisperLossOptions(implementation=implementation),
                suppressed_tokens=[10, 11],
            )
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1], atol=1e-5, rtol=1e-4)
    for original, optimized in zip(eager.parameters(), fast.parameters(), strict=True):
        if original.grad is not None:
            torch.testing.assert_close(
                original.grad, optimized.grad, atol=1e-5, rtol=1e-3
            )


def test_dataset_identity_ignores_derived_shuffle_cache(tmp_path):
    (tmp_path / "data.arrow").write_bytes(b"training rows")
    before = dataset_identity(tmp_path)
    (tmp_path / "cache-index.arrow").write_bytes(b"random permutation")
    assert dataset_identity(tmp_path) == before


def test_benchmark_pools_paired_clip_medians_and_preserves_wer():
    records = []
    for baseline, speculative in [([1, 2, 9], [2, 4, 12]), ([3, 4, 5], [1, 2, 3])]:
        records.append(
            {
                "baseline_generation_seconds": baseline,
                "speculative_generation_seconds": speculative,
                "generation_tokens": 10,
                "reference_text": "hello world",
                "teacher_transcript": "hello world",
                "speculative_transcript": "hello world",
                "decode_result": WhisperDecodeResult(
                    torch.tensor([[1, 2]]),
                    2,
                    draft_rounds=1,
                    proposed_tokens=3,
                    accepted_tokens=1,
                    proposed_by_position=[1, 1, 1],
                    accepted_by_position=[1, 0, 0],
                    eligible_by_position=[1, 1, 0],
                ),
            }
        )
    report = summarize_records(records, block_size=4)
    assert report["generation_speedup"] == 1.0  # (2 + 4) / (4 + 2)
    assert report["baseline_tokens_per_second"] == pytest.approx(20 / 6)
    assert report["mal"] == 2
    assert report["teacher_reference_error_rates"]["wer"] == 0
    assert (
        report["teacher_reference_error_rates"]
        == report["speculative_reference_error_rates"]
    )
    assert len(report["paired_bootstrap_speedup_95_percent_interval"]) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_draft_autocast_does_not_change_teacher_precision(monkeypatch):
    torch.manual_seed(30)
    teacher = tiny_teacher().cuda().bfloat16().eval().requires_grad_(False)
    teacher.generation_config.suppress_tokens = [teacher.config.eos_token_id]
    draft = build_whisper_draft(teacher, [0, 1]).cuda().float().eval()
    observed = []
    original_teacher = teacher.forward
    original_projection = draft.fc.forward

    def teacher_forward(*args, **kwargs):
        assert not torch.is_autocast_enabled("cuda")
        return original_teacher(*args, **kwargs)

    def draft_projection(*args, **kwargs):
        observed.append(torch.is_autocast_enabled("cuda"))
        return original_projection(*args, **kwargs)

    monkeypatch.setattr(teacher, "forward", teacher_forward)
    monkeypatch.setattr(draft.fc, "forward", draft_projection)
    speculative_whisper_decode(
        teacher,
        draft,
        torch.randn(1, 4, 16, device="cuda", dtype=torch.bfloat16),
        torch.tensor([[1, 3]], device="cuda"),
        max_new_tokens=8,
        draft_dtype=torch.bfloat16,
    )
    assert observed
    assert all(observed)


def test_evaluation_requires_explicit_opt_in_for_token_drift(monkeypatch):
    from speculators.train import whisper_eval  # noqa: PLC0415

    teacher = tiny_teacher().eval()
    draft = build_whisper_draft(teacher, [0, 1]).eval()
    sample = {
        "id": "clip",
        "split": "validation.clean",
        "audio": torch.randn(1, 4, 16),
        "prompt": torch.tensor([[1, 3]]),
        "tokens": torch.tensor([[1, 3, 4, 2]]),
        "reference_text": "",
        "duration": 1,
    }
    monkeypatch.setattr(
        whisper_eval,
        "speculative_whisper_decode",
        lambda *args, **kwargs: WhisperDecodeResult(
            torch.tensor([[1, 3, 5, 2]]),
            2,
            draft_rounds=1,
            proposed_tokens=3,
            accepted_tokens=0,
            proposed_by_position=[1, 1, 1],
            accepted_by_position=[0, 0, 0],
        ),
    )
    with pytest.raises(RuntimeError, match="Token mismatch"):
        whisper_eval.evaluate_samples(teacher, None, draft, [sample], max_new_tokens=4)
    report = whisper_eval.evaluate_samples(
        teacher,
        None,
        draft,
        [sample],
        max_new_tokens=4,
        strict_tokens=False,
    )
    assert not report["all_tokens_match"]
    assert report["token_match_rate"] == 0
    assert not report["samples"][0]["tokens_match"]
