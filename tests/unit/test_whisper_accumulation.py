"""Optimizer updates agree with a single target-weighted batch."""

import time
from types import SimpleNamespace

import pytest
import torch
import train_whisper_online as trainer

from speculators.train.whisper_online import prefetch_map


@pytest.mark.parametrize("accumulation", [2, 3])
def test_unequal_microbatches_match_single_batch(tmp_path, monkeypatch, accumulation):
    draft = torch.nn.Linear(1, 1, bias=False)
    draft.weight.data.fill_(0.025)
    optimizer = torch.optim.SGD(draft.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    values = [torch.tensor([[0.0]]), torch.tensor([[0.1], [0.2], [0.3]])]
    expected = draft.weight.detach().clone()
    # Both microbatches share the same parameter. Compare against their union.
    loss = ((draft.weight - torch.cat(values)) ** 2).mean()
    gradient = torch.autograd.grad(loss, draft.weight)[0]
    expected -= 0.1 * gradient.clamp(-1, 1)  # production gradient clipping

    def make_loss(*_args, **_kwargs):
        def compute(features):
            loss = ((draft.weight - features["values"]) ** 2).mean()
            count = torch.tensor(features["values"].numel(), dtype=torch.float32)
            return loss, {
                "weighted_loss_sum": loss.detach() * count,
                "weighted_loss_total": count,
                "eal_sum": count,
                "eal_total": count,
            }

        return compute

    monkeypatch.setattr(trainer, "make_whisper_loss", make_loss)
    monkeypatch.setattr(trainer, "loss_options", lambda _args: None)
    monkeypatch.setattr(
        trainer, "evaluate", lambda *_args, **_kwargs: {"validation_loss": 0}
    )
    monkeypatch.setattr(trainer, "snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer, "record_evaluation", lambda *_args, **_kwargs: None)
    args = SimpleNamespace(
        device="cpu",
        precision="float32",
        compile=False,
        max_anchors=4,
        batch_size=3,
        block_size=4,
        gradient_accumulation_steps=accumulation,
        benchmark_every=0,
        eval_every=10,
        decode_every=10,
        checkpoint_every=10,
        log_every=1,
        max_samples=4,
        output_dir=tmp_path,
    )
    teacher = SimpleNamespace(
        config=SimpleNamespace(max_target_positions=32),
        generation_config=SimpleNamespace(suppress_tokens=[]),
    )
    state = {
        "step": 0,
        "consumed": 0,
        "consumed_batches": 0,
        "skipped": {},
        "training_counts": {},
        "utterances": 0,
        "audio_seconds": 0,
        "valid_target_tokens": 0,
        "training_seconds": 0,
        "truncated_responses": 0,
        "loss_ema": None,
        "last_eval": -1,
        "best_eal": -1,
    }
    items = [
        {
            "features": {"values": value},
            "ready": None,
            "consumed": 1,
            "skipped": [],
            "sample_id": str(index),
            "utterances": 1,
            "audio_seconds": 1,
            "response_tokens": value.numel(),
            "truncated": 0,
            "teacher_feature_host_seconds": 0,
        }
        for index, value in enumerate(values)
    ]
    stream = prefetch_map(items, lambda value: value, capacity=0)
    trainer.run_updates(
        args,
        teacher,
        None,
        draft,
        optimizer=optimizer,
        scheduler=scheduler,
        items=stream,
        samples=[],
        state=state,
        metadata={"prepared_rows": 2},
        start=time.monotonic(),
        target=1,
    )
    torch.testing.assert_close(draft.weight, expected)
