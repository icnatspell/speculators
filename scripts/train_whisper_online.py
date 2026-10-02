"""Batched online Whisper distillation with isolated validation and exact resume."""

import argparse
import json
import random
import resource
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from functools import partial
from itertools import islice
from pathlib import Path

import torch
from datasets import load_from_disk
from train_whisper_dflash import (
    UnsupportedAudioError,
    audio_features,
    load_teacher_and_rows,
)

from speculators.data_generation.whisper import WhisperFeatureAdapter
from speculators.proposals.whisper import greedy_whisper_decode
from speculators.provenance import atomic_write
from speculators.train.utils import save_train_command
from speculators.train.whisper import (
    WhisperLossOptions,
    build_whisper_draft,
    load_whisper_draft,
    make_whisper_loss,
    metrics_to_cpu,
    whisper_autocast,
)
from speculators.train.whisper_eval import (
    collect_samples,
    evaluate_samples,
    validation_loss,
)
from speculators.train.whisper_online import (
    capture_rng,
    learning_rate_scale,
    prefetch_map,
    restore_rng,
    save_checkpoint,
    shuffled_dataset_epochs,
)
from speculators.train.whisper_runtime import (
    batched_rows,
    dataset_identity,
    hash_file,
    pack_whisper_features,
    precision_dtype,
    recover_checkpoint,
    verified_audio_bytes,
)


def parse_args(*, default_algorithm="dflash"):  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speculator-type", choices=["dflash", "eagle3"], default=default_algorithm
    )
    parser.add_argument(
        "--ttt-step-loss-decay", type=float, default=1.0, dest="rollout_decay"
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--teacher", default="openai/whisper-tiny.en")
    parser.add_argument("--teacher-revision", default="main")
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--dataset-config", default="all")
    parser.add_argument(
        "--dataset-revision", default="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
    )
    parser.add_argument("--split", nargs="+", default=["train.clean.100"])
    parser.add_argument(
        "--eval-split", nargs="+", default=["validation.clean", "validation.other"]
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument(
        "--max-anchors", type=int, default=32, help="DFlash anchor budget per utterance"
    )
    parser.add_argument(
        "--target-layer-ids",
        nargs="+",
        type=int,
        help="Default: three indices spanning decoder depth",
    )
    parser.add_argument("--num-draft-layers", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Utterances packed per drafter microbatch",
    )
    parser.add_argument(
        "--teacher-batch-size",
        type=int,
        help="Teacher microbatch; defaults to batch-size",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--bucket-buffer",
        type=int,
        default=0,
        help="Bounded token-length sorting window",
    )
    parser.add_argument("--audio-workers", type=int, default=1)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-device")
    parser.add_argument(
        "--prefetch-samples",
        type=int,
        default=0,
        help="Queued feature microbatches, not utterances",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="float32",
    )
    parser.add_argument(
        "--teacher-attention", choices=["eager", "sdpa"], default="sdpa"
    )
    parser.add_argument(
        "--draft-attention",
        choices=["eager", "sdpa", "simple_flex_attention"],
        default="eager",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--loss-implementation", choices=["eager", "fused"], default="eager"
    )
    parser.add_argument(
        "--raw-targets",
        action="store_true",
        help="Ablation: do not apply teacher suppression to loss",
    )
    parser.add_argument("--response-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--position-weight",
        choices=["fixed-exp-decay", "dpace"],
        default="fixed-exp-decay",
    )
    parser.add_argument("--decay-gamma", type=float, default=4.0)
    parser.add_argument("--dpace-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=128,
        help="Legacy; indexed training uses global shuffle",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1000,
        help="Fixed-anchor validation loss cadence",
    )
    parser.add_argument(
        "--decode-every", type=int, help="Dev MAL cadence; defaults to eval-every"
    )
    parser.add_argument(
        "--benchmark-every",
        type=int,
        default=0,
        help="Isolated timing cadence; 0 disables training benchmarks",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=100,
        help="Speaker/duration-balanced samples per dev split",
    )
    parser.add_argument("--eval-repetitions", type=int, default=2)
    parser.add_argument(
        "--cache-draft-context", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--response-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.cache_draft_context is None:
        args.cache_draft_context = args.speculator_type == "eagle3"
    args.teacher_batch_size = args.teacher_batch_size or args.batch_size
    args.decode_every = args.decode_every or args.eval_every
    positive = [
        "steps",
        "epochs",
        "batch_size",
        "teacher_batch_size",
        "gradient_accumulation_steps",
        "audio_workers",
        "max_anchors",
        "num_draft_layers",
        "checkpoint_every",
        "eval_every",
        "decode_every",
        "eval_samples",
        "eval_repetitions",
        "log_every",
        "learning_rate",
        "decay_gamma",
    ]
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("Counts, learning rate and decay-gamma must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max-samples must be positive")
    if (
        min(
            args.bucket_buffer,
            args.prefetch_samples,
            args.benchmark_every,
            args.response_ce_weight,
        )
        < 0
    ):
        parser.error("Buffer sizes, benchmark cadence and CE weight cannot be negative")
    if args.bucket_buffer and args.bucket_buffer < args.batch_size:
        parser.error("bucket-buffer must be at least batch-size")
    if not 0 <= args.minimum_lr_ratio <= 1 or not 0 <= args.warmup_steps < args.steps:
        parser.error("Invalid learning-rate schedule")
    if args.block_size < 2 or args.max_new_tokens <= args.block_size:  # noqa: PLR2004 -- anchor plus draft
        parser.error("Need block-size >= 2 and max-new-tokens > block-size")
    if (
        any(not split.startswith("validation.") for split in args.eval_split)
        and not args.synthetic
    ):
        parser.error(
            "Training evaluation uses validation splits; "
            "reserve tests for the standalone benchmark"
        )
    if len(args.split) != len(set(args.split)) or len(args.eval_split) != len(
        set(args.eval_split)
    ):
        parser.error("Training and evaluation splits must be unique")
    if args.synthetic and (
        args.device != "cpu"
        or args.teacher_device not in (None, "cpu")
        or args.prefetch_samples
    ):
        parser.error("Synthetic runs use CPU without threaded GPU prefetch")
    if not args.synthetic and (
        args.train_data is None or args.response_manifest is None
    ):
        parser.error("--train-data and --response-manifest are required")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        parser.error("stop-after must be within the schedule")
    if args.rollout_decay < 0:
        parser.error("ttt-step-loss-decay must be nonnegative")
    if args.speculator_type == "eagle3" and args.position_weight == "dpace":
        parser.error("EAGLE-3 uses --ttt-step-loss-decay, not DFlash Dpace weights")
    return args


def append_json(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def device_identity(device):
    device = torch.device(device)
    return device.type, (
        torch.cuda.current_device()
        if device.type == "cuda" and device.index is None
        else device.index
    )


def loss_options(args):
    return WhisperLossOptions(
        implementation=args.loss_implementation,
        policy_targets=not args.raw_targets,
        response_ce_weight=args.response_ce_weight,
        position_weight=args.position_weight,
        gamma=args.decay_gamma,
        dpace_alpha=args.dpace_alpha,
        rollout_decay=args.rollout_decay,
    )


@torch.no_grad()
def teacher_response(args, teacher, processor, row):
    if args.synthetic:
        # Feature production must not consume the trainer's global RNG.
        with torch.random.fork_rng():
            torch.manual_seed(args.seed + int(row) + 10000)
            audio = torch.randn(1, 4, 16)
        prompt = torch.tensor([[1, 3]])
    else:
        audio = audio_features(row, processor, teacher.device, dtype=teacher.dtype)
        prompt = torch.tensor(
            [processor.tokenizer.prefix_tokens], device=teacher.device
        )
    encoder = teacher.model.encoder(audio, return_dict=True)
    result = greedy_whisper_decode(
        teacher,
        audio,
        prompt,
        max_new_tokens=args.max_new_tokens,
        encoder_outputs=encoder,
    )
    return audio, prompt, encoder, result.tokens


def load_prepared_audio(args, processor, row):
    try:
        audio = audio_features(
            {
                "audio": {
                    "bytes": verified_audio_bytes(row, args.audio_root),
                    "path": None,
                }
            },
            processor,
            "cpu",
        )
        return row, audio
    except UnsupportedAudioError:
        return row, None


@torch.no_grad()
def prepare_training_item(args, processor, teacher, adapter, batch, *, audio_pool=None):
    started = time.monotonic()
    samples, skipped = [], []
    if args.synthetic:
        for row in batch:
            audio, prompt, _, tokens = teacher_response(args, teacher, processor, row)
            samples.append(
                (
                    {"id": str(row), "speaker_id": None},
                    audio,
                    tokens[0],
                    prompt.shape[1],
                    None,
                )
            )
    else:
        read = partial(load_prepared_audio, args, processor)
        loaded = audio_pool.map(read, batch) if audio_pool else map(read, batch)
        for row, audio in loaded:
            if audio is None:
                skipped.append("long_audio")
                continue
            if sum(row["loss_mask"]) < 2:  # noqa: PLR2004 -- anchor and at least one target
                skipped.append("short_response")
                continue
            samples.append(
                (
                    row,
                    audio,
                    torch.as_tensor(row["input_ids"]),
                    int(row["whisper_begin_index"]),
                    torch.as_tensor(row["loss_mask"]),
                )
            )
    feature_rows = []
    for start in range(0, len(samples), args.teacher_batch_size):
        chunk = samples[start : start + args.teacher_batch_size]
        audio = torch.cat([s[1] for s in chunk]).to(
            device=teacher.device, dtype=teacher.dtype
        )
        feature_rows.extend(
            adapter.extract_batch(
                audio,
                [s[2] for s in chunk],
                prompt_lengths=[s[3] for s in chunk],
                loss_masks=None if args.synthetic else [s[4] for s in chunk],
                pad_to_multiple=1 if args.synthetic else 16,
            )
        )
    packed = pack_whisper_features(feature_rows) if feature_rows else None
    ready = None
    if teacher.device.type == "cuda":
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(teacher.device))
    return {
        "features": packed,
        "ready": ready,
        "sample_id": ",".join(str(s[0]["id"]) for s in samples),
        "speaker_id": [s[0].get("speaker_id") for s in samples],
        "skipped": skipped,
        "consumed": len(batch),
        "utterances": len(samples),
        "response_tokens": sum(s[2].numel() - s[3] for s in samples),
        "audio_seconds": sum(
            float(s[0].get("audio_duration_seconds", 0)) for s in samples
        ),
        "truncated": sum(int(s[2][-1] != teacher.config.eos_token_id) for s in samples),
        "teacher_feature_host_seconds": time.monotonic() - started,
    }


def prepare_on_stream(function, batch, *, stream):
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        return function(batch)


def collect_validation(args, teacher, processor, metadata):
    rng = capture_rng()
    try:
        if args.synthetic:
            samples = []
            for row in range(args.eval_samples):
                audio, prompt, _, tokens = teacher_response(
                    args, teacher, processor, row
                )
                samples.append(
                    {
                        "id": str(row),
                        "split": "synthetic",
                        "audio": audio.cpu(),
                        "prompt": prompt.cpu(),
                        "tokens": tokens.cpu(),
                        "reference_text": "",
                        "duration": 0,
                    }
                )
            metadata["validation_coverage"] = {}
            return samples
        samples, coverage = collect_samples(
            args,
            teacher,
            processor,
            metadata["dataset_sha"],
            splits=args.eval_split,
            count=args.eval_samples,
            cache_path=args.output_dir / "validation_samples.pt",
        )
        metadata["validation_coverage"] = coverage
        return samples
    finally:
        restore_rng(rng)


@torch.no_grad()
def evaluate(
    args, teacher, processor, draft, samples, *, step, decode=True, benchmark=False
):
    rng = capture_rng()
    try:
        for device in {teacher.device, draft.embed_tokens.weight.device}:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        draft.eval()
        validation = validation_loss(
            teacher,
            draft,
            samples,
            max_anchors=args.max_anchors,
            options=loss_options(args),
            batch_size=args.teacher_batch_size,
            dtype=precision_dtype(args.precision, args.device),
            seed=args.seed,
        )
        report = {
            "step": step,
            "validation_loss": validation["loss"],
            "validation": validation,
            "coverage": args.validation_coverage,
        }
        if decode:
            report.update(
                evaluate_samples(
                    teacher,
                    processor,
                    draft,
                    samples,
                    max_new_tokens=args.max_new_tokens,
                    repetitions=args.eval_repetitions,
                    benchmark=benchmark,
                    cache_draft_context=args.cache_draft_context,
                    strict_tokens=teacher.dtype == torch.float32,
                    dtype=precision_dtype(args.precision, args.device),
                )
            )
        return report
    finally:
        restore_rng(rng)


def snapshot(args, draft, optimizer, scheduler, *, state, metadata, name):
    save_checkpoint(
        args.output_dir / name,
        draft,
        optimizer,
        scheduler,
        state=state,
        metadata={
            **metadata,
            "completed_steps": state["step"],
            "consumed_samples": state["consumed"],
            "best_eal": state["best_eal"],
            "skipped": state["skipped"],
        },
    )


def record_evaluation(args, report, metadata):
    directory = args.output_dir / "eval" / f"step-{report['step']:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    report["drafter_checkpoint_sha256"] = hash_file(
        args.output_dir / "latest" / "draft.safetensors"
    )
    report.update(
        teacher_sha=metadata["teacher_sha"], dataset_sha=metadata["dataset_sha"]
    )
    atomic_write(
        directory / "eval_command.txt",
        "# In-training evaluation\n"
        + (args.output_dir / "train_command.txt").read_text(),
    )
    shutil.copy2(args.output_dir / "speculators.patch", directory / "eval.patch")
    for invocation in args.output_dir.glob("resume-*"):
        if invocation.is_dir():
            shutil.copytree(
                invocation,
                directory / "provenance" / invocation.name,
                dirs_exist_ok=True,
            )
    atomic_write(
        directory / "drafter_checkpoint_sha256.txt",
        report["drafter_checkpoint_sha256"] + "\n",
    )
    atomic_write(directory / "results.json", json.dumps(report, indent=2))
    append_json(args.output_dir / "eval_metrics.jsonl", report)


def run_updates(  # noqa: C901 -- optimizer, logging and isolated validation loop
    args,
    teacher,
    processor,
    draft,
    *,
    optimizer,
    scheduler,
    items,
    samples,
    state,
    metadata,
    start,
    target,
    writer=None,
):
    dtype = precision_dtype(args.precision, args.device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.device(args.device).type == "cuda" and dtype == torch.float16,
    )
    if state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    function = make_whisper_loss(
        draft,
        options=loss_options(args),
        max_anchors=args.max_anchors,
        suppressed_tokens=teacher.generation_config.suppress_tokens or (),
        compile_model=args.compile,
    )
    parameters = [p for p in draft.parameters() if p.requires_grad]
    totals = {
        key: torch.tensor(value, device=args.device)
        for key, value in state.get("pending_metrics", {}).items()
    }
    iterator = iter(items)
    exhausted = False
    train_time = state.get("training_seconds", 0.0)
    while state["step"] < target and not exhausted:
        group = []
        target_count = torch.zeros((), device=args.device)
        # A fixed bound keeps scaled backward magnitudes small in FP16.
        target_bound = (
            args.gradient_accumulation_steps
            * args.batch_size
            * max(
                teacher.config.max_target_positions, args.max_anchors * args.block_size
            )
        )
        queue_wait = update_seconds = 0.0
        draft.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation_steps):
            wait_start = time.monotonic()
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                break
            state["consumed"] += item["consumed"]
            state["consumed_batches"] += 1
            for reason in item["skipped"]:
                state["skipped"][reason] += 1
            if item["ready"] is not None:
                item["ready"].synchronize()
            queue_wait += time.monotonic() - wait_start
            if item["features"] is None:
                continue
            update_start = time.monotonic()
            for value in item["features"].values():
                if value.device.type == "cuda":
                    value.record_stream(torch.cuda.current_stream(value.device))
            features = {
                key: value.to(args.device, non_blocking=True)
                for key, value in item.pop("features").items()
            }
            with (
                whisper_autocast(args.device, dtype),
                (
                    nullcontext()
                    if args.compile
                    else torch.compiler.set_stance("force_eager")
                ),
            ):
                loss, metrics = function(features)
            # clip_grad_norm_ checks finiteness once per optimizer update.
            count = metrics["weighted_loss_total"].detach()
            scaler.scale(loss * (count / target_bound)).backward()
            target_count += count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0) + value.detach()
            del features, loss, metrics
            item.pop("ready")
            group.append(
                item
            )  # metadata only: accumulation retains no hidden-state batches
            update_seconds += time.monotonic() - update_start
        if not group:
            continue
        update_start = time.monotonic()
        scaler.unscale_(optimizer)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(target_bound / target_count.clamp_min(1))
        torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=1.0, error_if_nonfinite=True
        )
        scaler.step(optimizer)
        scaler.update()
        state["scaler"] = scaler.state_dict()
        scheduler.step()
        state["step"] += 1
        update_seconds += time.monotonic() - update_start
        train_time += queue_wait + update_seconds
        state["training_seconds"] = train_time
        state["utterances"] += sum(i["utterances"] for i in group)
        state["valid_target_tokens"] += sum(i["response_tokens"] for i in group)
        state["audio_seconds"] += sum(i["audio_seconds"] for i in group)
        state["truncated_responses"] += sum(i["truncated"] for i in group)
        step = state["step"]
        do_eval = (
            bool(args.benchmark_every and step % args.benchmark_every == 0)
            or step % args.eval_every == 0
            or step % args.decode_every == 0
            or step == target
            or exhausted
        )
        if step == 1 or step % args.log_every == 0 or do_eval:
            counts = metrics_to_cpu(totals)
            totals = {}
            state["pending_metrics"] = {}
            for key, value in counts.items():
                state["training_counts"][key] = (
                    state["training_counts"].get(key, 0) + value
                )
            train_loss = counts["weighted_loss_sum"] / counts["weighted_loss_total"]
            state["loss_ema"] = (
                train_loss
                if state["loss_ema"] is None
                else 0.95 * state["loss_ema"] + 0.05 * train_loss
            )
            record = {
                "step": step,
                "sample_id": ",".join(i["sample_id"] for i in group),
                "loss": train_loss,
                "loss_ema": state["loss_ema"],
                "training_metrics": counts,
                "train_eal": counts["eal_sum"] / counts["eal_total"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "teacher_feature_host_seconds": sum(
                    i["teacher_feature_host_seconds"] for i in group
                ),
                "feature_queue_wait_seconds": queue_wait,
                "draft_update_seconds": update_seconds,
                "training_seconds": train_time,
                "elapsed_seconds": time.monotonic() - start,
                "utterances": state["utterances"],
                "consumed": state["consumed"],
                "audio_hours": state["audio_seconds"] / 3600,
                "utterances_per_second": state["utterances"] / train_time,
                "response_tokens_per_second": state["valid_target_tokens"] / train_time,
                "valid_targets_per_second": state["training_counts"].get(
                    "weighted_loss_total", 0
                )
                / train_time,
                "supervised_anchor_blocks": state["training_counts"].get(
                    "eal_total", 0
                ),
                "audio_hours_per_hour": state["audio_seconds"] / train_time,
                "completed_data_passes": state["consumed"]
                / metadata.get("prepared_rows", args.max_samples),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
            append_json(args.output_dir / "train_metrics.jsonl", record)
            atomic_write(
                args.output_dir / "progress.json",
                json.dumps({**record, "status": "running"}, indent=2),
            )
            if writer:
                for key in (
                    "loss",
                    "loss_ema",
                    "train_eal",
                    "learning_rate",
                    "utterances_per_second",
                    "valid_targets_per_second",
                ):
                    writer.add_scalar(f"train/{key}", record[key], step)
            print(
                f"step={step} loss={train_loss:.4f} EAL={record['train_eal']:.3f} "
                f"samples/s={record['utterances_per_second']:.2f}",
                flush=True,
            )
        if do_eval:
            # Wait for in-flight work, then hold production paused for ALL eval.
            with items.pause():
                report = evaluate(
                    args,
                    teacher,
                    processor,
                    draft,
                    samples,
                    step=step,
                    decode=(
                        step % args.decode_every == 0
                        or step == target
                        or exhausted
                        or bool(
                            args.benchmark_every and step % args.benchmark_every == 0
                        )
                    ),
                    benchmark=bool(
                        args.benchmark_every and step % args.benchmark_every == 0
                    ),
                )
                state["last_eval"] = step
                if report.get("eal") is not None and report["eal"] > state["best_eal"]:
                    state["best_eal"] = report["eal"]
                    snapshot(
                        args,
                        draft,
                        optimizer,
                        scheduler,
                        state=state,
                        metadata=metadata,
                        name="best",
                    )
                snapshot(
                    args,
                    draft,
                    optimizer,
                    scheduler,
                    state=state,
                    metadata=metadata,
                    name="latest",
                )
                record_evaluation(args, report, metadata)
            if writer:
                writer.add_scalar("validation/loss", report["validation_loss"], step)
                for split, values in report.get("by_split", {}).items():
                    writer.add_scalar(f"validation/{split}/mal", values["mal"], step)
                    wer = values["teacher_reference_error_rates"]["wer"]
                    if wer is not None:
                        writer.add_scalar(f"validation/{split}/wer", wer, step)
            print(
                f"eval step={step} val_loss={report['validation_loss']:.4f} "
                f"MAL={report.get('mal')}",
                flush=True,
            )
        elif step % args.checkpoint_every == 0:
            with items.pause():
                state["pending_metrics"] = metrics_to_cpu(totals) if totals else {}
                snapshot(
                    args,
                    draft,
                    optimizer,
                    scheduler,
                    state=state,
                    metadata=metadata,
                    name="latest",
                )
        del group
    state["pending_metrics"] = metrics_to_cpu(totals) if totals else {}
    if state["step"] and state["last_eval"] != state["step"]:
        with items.pause():
            report = evaluate(
                args, teacher, processor, draft, samples, step=state["step"]
            )
            state["last_eval"] = state["step"]
            if report.get("eal") is not None and report["eal"] > state["best_eal"]:
                state["best_eal"] = report["eal"]
                snapshot(
                    args,
                    draft,
                    optimizer,
                    scheduler,
                    state=state,
                    metadata=metadata,
                    name="best",
                )
            snapshot(
                args,
                draft,
                optimizer,
                scheduler,
                state=state,
                metadata=metadata,
                name="latest",
            )
            record_evaluation(args, report, metadata)


def load_resume(latest, metadata, device, optimizer, scheduler, *, state):
    previous = json.loads((latest / "results.json").read_text())
    mutable = {
        "output_dir",
        "resume",
        "stop_after",
        "device",
        "teacher_device",
        "audio_root",
        "train_data",
        "response_manifest",
        "prefetch_samples",
        "audio_workers",
        "tensorboard",
        "validation_coverage",
        "validation_ids",
    }
    for key, value in metadata.items():
        legacy_defaults = {"speculator_type": "dflash", "rollout_decay": 1.0}
        if key not in mutable and previous.get(key, legacy_defaults.get(key)) != value:
            raise ValueError(f"Resume configuration changed: {key}")
    saved = torch.load(
        latest / "trainer_state.pt", map_location=device, weights_only=False
    )
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    state.update({key: saved[key] for key in state if key in saved})
    return saved, state


def main(*, default_algorithm="dflash"):  # noqa: C901
    args = parse_args(default_algorithm=default_algorithm)
    args.teacher_device = args.teacher_device or args.device
    if args.synthetic:
        args.teacher_device = "cpu"
        args.max_samples = args.max_samples or 10000
        torch.set_num_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError("Use a fresh run directory, or --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest = (
        recover_checkpoint(args.output_dir / "latest")
        if args.resume
        else args.output_dir / "latest"
    )
    save_train_command(
        str(
            args.output_dir / f"resume-{time.time_ns()}"
            if args.resume
            else args.output_dir
        )
    )
    splits = args.split
    args.split = splits[0]
    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if args.synthetic:
        metadata.update(teacher_sha="synthetic", dataset_sha="synthetic")
    args.skip_source_rows = not args.synthetic
    teacher, processor, source_rows = load_teacher_and_rows(args, metadata)
    args.split = splits
    args.teacher_revision = metadata["teacher_sha"]
    if args.target_layer_ids is None:
        depth = teacher.config.decoder_layers
        args.target_layer_ids = sorted({0, depth // 2, depth})
    args.dtype = precision_dtype(args.precision, args.device)
    if args.dtype == torch.float16 and torch.device(args.device).type != "cuda":
        raise ValueError("FP16 training requires CUDA")
    metadata.update(
        split=splits,
        teacher_revision=args.teacher_revision,
        target_layer_ids=args.target_layer_ids,
        training_policy="prepared_greedy_teacher_responses_online_teacher_features",
        accumulation_policy="supervised-target-weighted-v1",
    )
    if args.prefetch_samples and device_identity(teacher.device) == device_identity(
        args.device
    ):
        raise ValueError(
            "Feature prefetch requires distinct teacher and drafter devices"
        )
    if args.synthetic:
        rows = source_rows
        metadata.update(prepared_identity="synthetic", response_manifest_sha256=None)
    else:
        manifest = json.loads(args.response_manifest.read_text())
        for key, value in {
            "teacher_sha": metadata["teacher_sha"],
            "dataset_sha": metadata["dataset_sha"],
            "splits": splits,
            "prompt_token_ids": list(processor.tokenizer.prefix_tokens),
        }.items():
            if (
                manifest.get(key, [manifest.get("split")] if key == "splits" else None)
                != value
            ):
                raise ValueError(f"Response manifest {key} does not match training")
        if manifest.get("status", "complete") != "complete":
            raise ValueError("Response generation is incomplete")
        prepared = load_from_disk(str(args.train_data))
        required = {
            "input_ids",
            "loss_mask",
            "audio_url",
            "audio_sha256",
            "whisper_begin_index",
            "source_split",
            "id",
        }
        if not required <= set(prepared.column_names):
            raise ValueError("Prepared Whisper metadata columns are missing")
        if set(prepared.unique("source_split")) != set(splits):
            raise ValueError(
                "Prepared source splits do not match the response manifest"
            )
        prep_manifest = args.train_data / "preparation_manifest.json"
        if prep_manifest.exists():
            source_hashes = json.loads(prep_manifest.read_text())["sources"].values()
            if manifest["output_sha256"] not in source_hashes:
                raise ValueError(
                    "Prepared rows are not bound to the specified response manifest"
                )
        else:
            raise ValueError("Re-run prepare-data to record its source identity")
        provenance_root = args.output_dir / "data_provenance"
        for label, source_dir, names in (
            (
                "generation",
                args.response_manifest.parent,
                (
                    args.response_manifest.name,
                    "generation_command.txt",
                    "speculators.patch",
                ),
            ),
            (
                "preparation",
                args.train_data,
                (
                    "preparation_manifest.json",
                    "prepare_command.txt",
                    "speculators.patch",
                ),
            ),
        ):
            destination = provenance_root / label
            destination.mkdir(parents=True, exist_ok=True)
            for name in names:
                if (source_dir / name).exists():
                    shutil.copy2(source_dir / name, destination / name)
            if label == "generation":
                for invocation in source_dir.glob("resume-*"):
                    if invocation.is_dir():
                        shutil.copytree(
                            invocation,
                            destination / invocation.name,
                            dirs_exist_ok=True,
                        )
        metadata.update(
            prepared_identity=dataset_identity(args.train_data),
            response_manifest_sha256=hash_file(args.response_manifest),
            prepared_rows=len(prepared),
        )
        args.max_samples = args.max_samples or len(prepared) * args.epochs
        rows = shuffled_dataset_epochs(
            prepared,
            epochs=args.epochs,
            seed=args.seed,
            buffer_size=args.shuffle_buffer,
        )
    metadata["max_samples"] = args.max_samples
    batches = batched_rows(
        islice(rows, args.max_samples),
        args.batch_size,
        bucket_buffer=0 if args.synthetic else args.bucket_buffer,
    )
    teacher.eval().requires_grad_(False)
    adapter = WhisperFeatureAdapter(teacher, args.target_layer_ids)
    draft = (
        (
            load_whisper_draft(latest)
            if args.resume
            else build_whisper_draft(
                teacher,
                args.target_layer_ids,
                block_size=args.block_size,
                num_layers=args.num_draft_layers,
                attention_implementation=args.draft_attention,
                algorithm=args.speculator_type,
            )
        )
        .to(args.device)
        .float()
    )
    optimizer = torch.optim.AdamW(
        [p for p in draft.parameters() if p.requires_grad],
        lr=args.learning_rate,
        fused=torch.device(args.device).type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        partial(
            learning_rate_scale,
            total_steps=args.steps,
            warmup_steps=args.warmup_steps,
            minimum_ratio=args.minimum_lr_ratio,
        ),
    )
    state = {
        "step": 0,
        "consumed": 0,
        "consumed_batches": 0,
        "skipped": {"long_audio": 0, "short_response": 0},
        "truncated_responses": 0,
        "best_eal": -1.0,
        "last_eval": -1,
        "training_counts": {},
        "utterances": 0,
        "audio_seconds": 0.0,
        "valid_target_tokens": 0,
        "training_seconds": 0.0,
        "loss_ema": None,
        "scaler": {},
        "pending_metrics": {},
    }
    saved = None
    if args.resume:
        saved, state = load_resume(
            latest, metadata, args.device, optimizer, scheduler, state=state
        )
    samples = collect_validation(args, teacher, processor, metadata)
    args.validation_coverage = metadata["validation_coverage"]
    metadata["validation_ids"] = [sample["id"] for sample in samples]
    atomic_write(args.output_dir / "run_config.json", json.dumps(metadata, indent=2))
    if saved:
        replayed = sum(1 for _ in islice(batches, state["consumed_batches"]))
        if replayed != state["consumed_batches"]:
            raise RuntimeError("Could not recover exact microbatch position")
        restore_rng(saved["rng"])
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        writer = SummaryWriter(
            str(args.output_dir / "tensorboard"),
            purge_step=state["step"] + 1 if saved else None,
        )
    started = time.monotonic()
    try:
        if state["last_eval"] < 0:
            report = evaluate(args, teacher, processor, draft, samples, step=0)
            state["last_eval"] = 0
            state["best_eal"] = report.get("eal") or -1.0
            snapshot(
                args,
                draft,
                optimizer,
                scheduler,
                state=state,
                metadata=metadata,
                name="best",
            )
            snapshot(
                args,
                draft,
                optimizer,
                scheduler,
                state=state,
                metadata=metadata,
                name="latest",
            )
            record_evaluation(args, report, metadata)
        with ThreadPoolExecutor(max_workers=args.audio_workers) as pool:
            prepare = partial(
                prepare_training_item,
                args,
                processor,
                teacher,
                adapter,
                audio_pool=pool,
            )
            stream = (
                torch.cuda.Stream(device=teacher.device)
                if args.prefetch_samples
                else None
            )
            items = prefetch_map(
                batches,
                partial(prepare_on_stream, prepare, stream=stream),
                capacity=args.prefetch_samples,
            )
            with closing(items):
                run_updates(
                    args,
                    teacher,
                    processor,
                    draft,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    items=items,
                    samples=samples,
                    state=state,
                    metadata=metadata,
                    start=started,
                    target=args.stop_after or args.steps,
                    writer=writer,
                )
        snapshot(
            args,
            draft,
            optimizer,
            scheduler,
            state=state,
            metadata=metadata,
            name="latest",
        )
    finally:
        if writer:
            writer.close()
    outcome = {
        **metadata,
        **state,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": sum(
            torch.cuda.max_memory_allocated(d)
            for d in {teacher.device, draft.embed_tokens.weight.device}
            if d.type == "cuda"
        ),
        "status": "complete"
        if state["step"] == args.steps
        else "stopped"
        if state["step"] == args.stop_after
        else "data_exhausted",
    }
    atomic_write(args.output_dir / "results.json", json.dumps(outcome, indent=2))
    print(f"Training {outcome['status']} at step {state['step']}", flush=True)


if __name__ == "__main__":
    main()
