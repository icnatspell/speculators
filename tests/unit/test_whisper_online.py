import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file

from speculators.proposals.whisper import WhisperDecodeResult
from speculators.train.whisper_online import (
    acceptance_metrics,
    corpus_error_rates,
    learning_rate_scale,
    prefetch_map,
    shuffled_dataset_epochs,
)


def test_schedule_warms_up_and_decays():
    scales = [
        learning_rate_scale(step, total_steps=10, warmup_steps=2, minimum_ratio=0.1)
        for step in range(11)
    ]
    assert scales[:3] == [0.5, 1.0, 1.0]
    assert scales[-1] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(scales[2:], scales[3:], strict=False))


def test_shuffled_dataset_epochs_are_repeatable_and_distinct():
    dataset = Dataset.from_dict({"sample": list(range(12))})
    first = list(shuffled_dataset_epochs(dataset, epochs=2, seed=7, buffer_size=12))
    repeated = list(shuffled_dataset_epochs(dataset, epochs=2, seed=7, buffer_size=12))
    epoch_one = [row["sample"] for row in first[:12]]
    epoch_two = [row["sample"] for row in first[12:]]
    assert first == repeated
    assert sorted(epoch_one) == sorted(epoch_two) == list(range(12))
    assert epoch_one != epoch_two


def test_reference_error_rates_normalize_case_and_punctuation():
    rates = corpus_error_rates(
        ["Hello, world!", "Red blue"], ["hello world", "Red glue"]
    )
    assert rates["wer"] == pytest.approx(1 / 4)
    assert rates["cer"] == pytest.approx(1 / 17)


def test_prefetch_map_overlaps_production_and_preserves_order():
    produced_second = threading.Event()

    def prepare(item):
        if item == 1:
            produced_second.set()
        return item * 2

    items = prefetch_map(range(4), prepare, capacity=1)
    try:
        assert next(items) == 0
        assert produced_second.wait(timeout=1)
        assert list(items) == [2, 4, 6]
    finally:
        items.close()


def test_prefetch_map_propagates_producer_errors():
    def fail(item):
        if item == 1:
            raise ValueError("feature generation failed")
        return item

    items = prefetch_map(range(3), fail, capacity=1)
    with pytest.raises(ValueError, match="feature generation failed"):
        list(items)


def test_eal_pools_rounds_and_includes_bonus():
    results = [
        WhisperDecodeResult(
            torch.tensor([[1]]),
            2,
            proposed_tokens=3,
            accepted_tokens=3,
            draft_rounds=1,
            proposed_by_position=[1, 1, 1],
            accepted_by_position=[1, 1, 1],
        ),
        WhisperDecodeResult(
            torch.tensor([[1]]),
            4,
            proposed_tokens=9,
            accepted_tokens=0,
            draft_rounds=3,
            proposed_by_position=[3, 3, 3],
            accepted_by_position=[0, 0, 0],
        ),
    ]
    metrics = acceptance_metrics(results, block_size=4)
    assert metrics["accepted_draft_length"] == 0.75
    assert metrics["eal"] == metrics["mal"] == 1.75
    assert metrics["acceptance_rate"] == 0.25
    assert metrics["proposed_by_position"] == [4, 4, 4]
    assert metrics["accepted_by_position"] == [1, 1, 1]
    assert metrics["acceptance_by_position"] == [0.25] * 3


@pytest.mark.parametrize(("batch_size", "accumulation"), [(1, 1), (2, 2)])
def test_online_training_resume_matches_uninterrupted(
    tmp_path, batch_size, accumulation
):
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": f"{root / 'src'}:{root / 'hs_connectors/src'}",
        "OMP_NUM_THREADS": "1",
    }
    command = [
        sys.executable,
        str(root / "scripts/train_whisper_dflash_online.py"),
        "--synthetic",
        "--device",
        "cpu",
        "--steps",
        "6",
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation-steps",
        str(accumulation),
        "--bucket-buffer",
        "8",
        "--max-samples",
        "30",
        "--max-new-tokens",
        "16",
        "--max-anchors",
        "4",
        "--warmup-steps",
        "1",
        "--eval-samples",
        "2",
        "--eval-repetitions",
        "1",
        "--checkpoint-every",
        "3",
        "--eval-every",
        "3",
    ]
    resumed, full = tmp_path / "resumed", tmp_path / "full"
    for output, extra in [
        (resumed, ["--stop-after", "3"]),
        (resumed, ["--resume"]),
        (full, []),
    ]:
        subprocess.run(  # noqa: S603 -- fixed local script and test-controlled argv
            [*command, "--output-dir", str(output), *extra],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    a, b = (
        load_file(str(directory / "latest/draft.safetensors"))
        for directory in [resumed, full]
    )
    assert all(torch.equal(a[key], b[key]) for key in a)
    logs = [
        [
            json.loads(line)
            for line in (directory / "train_metrics.jsonl").read_text().splitlines()
        ]
        for directory in [resumed, full]
    ]
    for key in ["sample_id", "loss", "learning_rate", "train_eal"]:
        assert [row[key] for row in logs[0]] == [row[key] for row in logs[1]]
    assert json.loads((resumed / "results.json").read_text())["status"] == "complete"
    assert sorted(path.name for path in resumed.glob("latest*")) == ["latest"]
    assert sorted(path.name for path in resumed.glob("best*")) == ["best"]
    for directory in [resumed / "latest", resumed / "best"]:
        assert (directory / "train_command.txt").exists()
        assert (directory / "speculators.patch").exists()
        assert (directory / "trainer_state.pt").exists()
