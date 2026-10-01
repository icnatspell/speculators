"""Train a Whisper draft from prepared teacher responses without stored features."""

import argparse
import hashlib
import json
import random
import resource
import shutil
import time
from contextlib import closing, nullcontext
from functools import partial
from itertools import islice
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
from datasets import load_from_disk
from train_whisper_dflash import (
    UnsupportedAudioError,
    audio_features,
    load_teacher_and_rows,
)

from speculators.data_generation.whisper import WhisperFeatureAdapter
from speculators.proposals.whisper import (
    greedy_whisper_decode,
    speculative_whisper_decode,
)
from speculators.provenance import atomic_write
from speculators.train.utils import save_train_command
from speculators.train.whisper import (
    build_whisper_draft,
    load_whisper_draft,
    train_whisper_step,
)
from speculators.train.whisper_online import (
    acceptance_metrics,
    corpus_error_rates,
    learning_rate_scale,
    restore_rng,
    save_checkpoint,
    shuffled_dataset_epochs,
)


def parse_args():  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--teacher", default="openai/whisper-tiny.en")
    parser.add_argument(
        "--teacher-revision", default="87c7102498dcde7456f24cfd30239ca606ed9063"
    )
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--dataset-config", default="all")
    parser.add_argument(
        "--dataset-revision", default="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["train.clean.100"],
        help="Training source split(s) represented in the prepared data.",
    )
    parser.add_argument(
        "--eval-split",
        nargs="+",
        default=["validation.clean", "validation.other"],
        help="Held-out split(s); eval-samples are collected from each split.",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--max-anchors", type=int, default=32)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--eval-repetitions", type=int, default=2)
    parser.add_argument(
        "--train-data",
        type=Path,
        help="Prepared dataset produced by speculators prepare-data.",
    )
    parser.add_argument(
        "--response-manifest",
        type=Path,
        help="Generation manifest paired with --train-data.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Stop cleanly at this update without changing the schedule.",
    )
    args = parser.parse_args()
    if (
        min(
            args.steps,
            args.max_anchors,
            args.eval_samples,
            args.eval_repetitions,
            args.checkpoint_every,
            args.eval_every,
            args.learning_rate,
        )
        <= 0
    ):
        parser.error("Counts and learning rate must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max-samples must be positive when provided")
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if (
        args.shuffle_buffer < 1
        or not 0 <= args.minimum_lr_ratio <= 1
        or not 0 <= args.warmup_steps < args.steps
    ):
        parser.error("Invalid shuffle buffer or learning-rate schedule")
    if args.block_size < 2 or args.max_new_tokens <= args.block_size:  # noqa: PLR2004
        parser.error("Need block-size >= 2 and max-new-tokens > block-size")
    if any(split.startswith("train") for split in args.eval_split):
        parser.error("Evaluation must use held-out audio")
    if len(args.split) != len(set(args.split)) or len(args.eval_split) != len(
        set(args.eval_split)
    ):
        parser.error("Training and evaluation splits must be unique")
    if args.synthetic and args.device != "cpu":
        parser.error("Synthetic runs use CPU")
    if not args.synthetic and args.train_data is None:
        parser.error("--train-data is required unless --synthetic is set")
    if not args.synthetic and args.response_manifest is None:
        parser.error("--response-manifest is required unless --synthetic is set")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        parser.error("stop-after must be within the training schedule")
    return args


def append_json(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


@torch.no_grad()
def teacher_response(args, teacher, processor, row):
    if args.synthetic:
        audio = torch.randn(1, 4, 16)
        prompt = torch.tensor([[1, 3]])
    else:
        audio = audio_features(row, processor, teacher.device)
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


def prepared_teacher_sample(row, processor, teacher):
    """Load one prepared audio clip and its fixed teacher transcript."""
    parsed = urlparse(row["audio_url"])
    if parsed.scheme != "file":
        raise ValueError("Prepared Whisper audio_url must use the file:// scheme")
    audio = audio_features(
        {"audio": {"bytes": None, "path": unquote(parsed.path)}},
        processor,
        teacher.device,
    )
    tokens = torch.as_tensor(row["input_ids"], device=teacher.device).view(1, -1)
    loss_mask = torch.as_tensor(row["loss_mask"], device=teacher.device).view(1, -1)
    prompt_length = int(row["whisper_begin_index"])
    encoder = teacher.model.encoder(audio, return_dict=True)
    return audio, tokens, prompt_length, loss_mask, encoder


def collect_validation(args, teacher, processor, metadata):
    if args.synthetic:
        split_rows = [("synthetic", iter(range(args.eval_samples)))]
    else:
        from train_whisper_dflash import load_rows  # noqa: PLC0415

        split_rows = [
            (split, load_rows(args, split, metadata["dataset_sha"]))
            for split in args.eval_split
        ]
    samples = []
    for split, rows in split_rows:
        split_count = 0
        with nullcontext(rows) if args.synthetic else closing(rows):
            for row in rows:
                try:
                    audio, prompt, encoder, tokens = teacher_response(
                        args, teacher, processor, row
                    )
                except UnsupportedAudioError:
                    continue
                samples.append(
                    {
                        "id": str(row) if args.synthetic else row["id"],
                        "split": split,
                        "reference_text": "" if args.synthetic else row.get("text", ""),
                        "audio": audio.cpu(),
                        "prompt": prompt.cpu(),
                        "tokens": tokens.cpu(),
                    }
                )
                if not args.synthetic and not metadata.get(
                    "teacher_policy_hf_verified"
                ):
                    hf_tokens = teacher.generate(
                        input_features=audio,
                        decoder_input_ids=prompt,
                        do_sample=False,
                        num_beams=1,
                        return_timestamps=False,
                        max_new_tokens=args.max_new_tokens,
                        return_dict_in_generate=True,
                    ).sequences
                    if not torch.equal(hf_tokens, tokens):
                        raise RuntimeError(
                            "Teacher response policy differs from HF generation"
                        )
                    metadata["teacher_policy_hf_verified"] = True
                del encoder
                split_count += 1
                if split_count == args.eval_samples:
                    break
        if split_count != args.eval_samples:
            raise RuntimeError(f"Insufficient held-out clips in {split}")
    return samples


@torch.no_grad()
def evaluate(args, teacher, processor, draft, samples, step):  # noqa: PLR0917
    import statistics  # noqa: PLC0415

    records, results = [], []
    references, hypotheses = [], []
    baseline_seconds = speculative_seconds = 0.0
    draft.eval()
    for sample in samples:
        audio, prompt = (
            sample["audio"].to(teacher.device),
            sample["prompt"].to(teacher.device),
        )
        encoder = teacher.model.encoder(audio, return_dict=True)
        shared = {
            "max_new_tokens": args.max_new_tokens,
            "encoder_outputs": encoder,
            "measure_generation": True,
        }
        functions = {
            "baseline": partial(
                greedy_whisper_decode, teacher, audio, prompt, **shared
            ),
            "speculative": partial(
                speculative_whisper_decode, teacher, draft, audio, prompt, **shared
            ),
        }
        elapsed = {name: [] for name in functions}
        for fn in functions.values():
            if not torch.equal(fn().tokens.cpu(), sample["tokens"]):
                raise RuntimeError(f"Warmup token mismatch on {sample['id']}")
        for repeat in range(args.eval_repetitions):
            order = (
                ["baseline", "speculative"]
                if repeat % 2 == 0
                else ["speculative", "baseline"]
            )
            for name in order:
                result = functions[name]()
                if not torch.equal(result.tokens.cpu(), sample["tokens"]):
                    raise RuntimeError(f"Evaluation token mismatch on {sample['id']}")
                elapsed[name].append(result.generation_seconds)
                if name == "speculative" and repeat == 0:
                    results.append(result)
        base, spec = (
            statistics.median(elapsed[name]) for name in ["baseline", "speculative"]
        )
        baseline_tokens = functions["baseline"]().tokens
        hypothesis = (
            ""
            if args.synthetic
            else processor.tokenizer.batch_decode(
                baseline_tokens, skip_special_tokens=True
            )[0]
        )
        references.append(sample["reference_text"])
        hypotheses.append(hypothesis)
        baseline_seconds += base
        speculative_seconds += spec
        records.append(
            {
                "sample_id": sample["id"],
                "split": sample["split"],
                "reference_text": sample["reference_text"],
                "teacher_transcript": hypothesis,
                "generation_tokens": max(
                    0, sample["tokens"].shape[1] - prompt.shape[1] - 1
                ),
                "baseline_generation_seconds": elapsed["baseline"],
                "speculative_generation_seconds": elapsed["speculative"],
                "tokens_match": True,
            }
        )
    return {
        "step": step,
        **acceptance_metrics(results, block_size=args.block_size),
        "baseline_generation_seconds": baseline_seconds,
        "speculative_generation_seconds": speculative_seconds,
        "generation_speedup": baseline_seconds / speculative_seconds
        if speculative_seconds
        else None,
        "timing_scope": "first_token_available_to_last_token_available",
        "samples": records,
        "all_tokens_match": True,
        "teacher_reference_error_rates": corpus_error_rates(references, hypotheses),
    }


def snapshot(args, draft, optimizer, scheduler, state, metadata, name):  # noqa: PLR0917
    meta = {
        **metadata,
        "completed_steps": state["step"],
        "consumed_samples": state["consumed"],
        "skipped": state["skipped"],
        "best_eal": state["best_eal"],
    }
    save_checkpoint(
        args.output_dir / name, draft, optimizer, scheduler, state=state, metadata=meta
    )
    digest = hashlib.sha256(
        (args.output_dir / name / "draft.safetensors").read_bytes()
    ).hexdigest()
    atomic_write(
        args.output_dir / name / "drafter_checkpoint_sha256.txt", digest + "\n"
    )


def record_evaluation(args, report, metadata):
    directory = args.output_dir / "eval" / f"step-{report['step']:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    report["drafter_checkpoint_sha256"] = (
        (args.output_dir / "latest" / "drafter_checkpoint_sha256.txt")
        .read_text()
        .strip()
    )
    report["teacher_sha"] = metadata["teacher_sha"]
    report["dataset_sha"] = metadata["dataset_sha"]
    report["splits"] = args.eval_split
    command = (args.output_dir / "train_command.txt").read_text()
    atomic_write(
        directory / "eval_command.txt",
        f"# In-training evaluation at update {report['step']}\n{command}",
    )
    shutil.copy2(args.output_dir / "speculators.patch", directory / "eval.patch")
    atomic_write(
        directory / "drafter_checkpoint_sha256.txt",
        report["drafter_checkpoint_sha256"] + "\n",
    )
    atomic_write(directory / "results.json", json.dumps(report, indent=2))
    append_json(args.output_dir / "eval_metrics.jsonl", report)


def run_updates(  # noqa: C901
    args,
    teacher,
    processor,
    draft,
    adapter,
    *,
    optimizer,
    scheduler,
    rows,
    samples,
    state,
    metadata,
    start,
    target,
):
    for row in rows:
        if state["step"] >= target or state["consumed"] >= args.max_samples:
            break
        state["consumed"] += 1
        sample_id = (
            str(row)
            if args.synthetic
            else str(row.get("id", f"sample-{state['consumed']}"))
        )
        try:
            if args.synthetic:
                audio, prompt, encoder, tokens = teacher_response(
                    args, teacher, processor, row
                )
                prompt_length = prompt.shape[1]
                loss_mask = None
            else:
                audio, tokens, prompt_length, loss_mask, encoder = (
                    prepared_teacher_sample(row, processor, teacher)
                )
                prompt = tokens[:, :prompt_length]
        except UnsupportedAudioError:
            state["skipped"]["long_audio"] += 1
            append_json(
                args.output_dir / "train_metrics.jsonl",
                {
                    "sample_id": sample_id,
                    "skip": "long_audio",
                    "consumed": state["consumed"],
                },
            )
            continue
        response_tokens = (
            int(loss_mask.sum().item())
            if loss_mask is not None
            else tokens.shape[1] - prompt.shape[1]
        )
        if response_tokens <= args.block_size:
            state["skipped"]["short_response"] += 1
            del audio, prompt, encoder, tokens
            continue
        truncated = tokens[0, -1].item() != teacher.config.eos_token_id
        state["truncated_responses"] += int(truncated)
        features = adapter.extract(
            audio,
            tokens,
            prompt_length=prompt_length,
            loss_mask=loss_mask,
            encoder_outputs=encoder,
        )
        lr = optimizer.param_groups[0]["lr"]
        training_metrics = {}
        loss = train_whisper_step(
            draft,
            optimizer,
            features,
            max_anchors=args.max_anchors,
            metrics_sink=training_metrics,
        )
        for key, value in training_metrics.items():
            state["training_counts"][key] = (
                state["training_counts"].get(key, 0.0) + value
            )
        scheduler.step()
        state["step"] += 1
        record = {
            "step": state["step"],
            "sample_id": sample_id,
            "speaker_id": None if args.synthetic else row.get("speaker_id"),
            "loss": loss,
            "training_metrics": training_metrics,
            "train_eal": training_metrics["eal_sum"] / training_metrics["eal_total"],
            "learning_rate": lr,
            "response_tokens": response_tokens,
            "truncated": truncated,
            "consumed": state["consumed"],
            "elapsed_seconds": time.monotonic() - start,
            "cuda_allocated_bytes": torch.cuda.memory_allocated()
            if teacher.device.type == "cuda"
            else 0,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        append_json(args.output_dir / "train_metrics.jsonl", record)
        del audio, prompt, encoder, tokens, features
        if state["step"] % 25 == 0:
            atomic_write(
                args.output_dir / "progress.json",
                json.dumps(
                    {
                        **record,
                        "status": "running",
                        "skipped": state["skipped"],
                        "truncated_responses": state["truncated_responses"],
                    },
                    indent=2,
                ),
            )
            train_eal = (
                state["training_counts"]["eal_sum"]
                / state["training_counts"]["eal_total"]
            )
            print(
                f"step={state['step']} loss={loss:.4f} lr={lr:.6g} "
                f"elapsed={record['elapsed_seconds']:.1f}s "
                f"train_EAL={train_eal:.3f}",
                flush=True,
            )
        if state["step"] % args.eval_every == 0 or state["step"] == target:
            report = evaluate(args, teacher, processor, draft, samples, state["step"])
            state["last_eval"] = state["step"]
            if report["eal"] > state["best_eal"]:
                state["best_eal"] = report["eal"]
                snapshot(args, draft, optimizer, scheduler, state, metadata, "best")
            snapshot(args, draft, optimizer, scheduler, state, metadata, "latest")
            record_evaluation(args, report, metadata)
            print(
                f"eval step={state['step']} EAL={report['eal']:.3f} "
                f"MAL={report['mal']} acceptance={report['acceptance_rate']} "
                f"speedup={report['generation_speedup']}",
                flush=True,
            )
        if (
            state["step"] % args.checkpoint_every == 0 or state["step"] == target
        ) and state["last_eval"] != state["step"]:
            snapshot(args, draft, optimizer, scheduler, state, metadata, "latest")


def load_resume(latest, metadata, device, optimizer, scheduler, state):  # noqa: PLR0917
    previous = json.loads((latest / "results.json").read_text())
    for key in (
        "teacher_sha",
        "dataset_sha",
        "train_data",
        "response_manifest_sha256",
        "split",
        "max_samples",
        "epochs",
        "seed",
        "shuffle_buffer",
        "steps",
        "learning_rate",
        "warmup_steps",
        "minimum_lr_ratio",
        "max_new_tokens",
        "max_anchors",
        "block_size",
        "target_layer_ids",
        "eval_split",
        "eval_samples",
        "eval_repetitions",
    ):
        if previous[key] != metadata[key]:
            raise ValueError(f"Resume configuration changed: {key}")
    saved = torch.load(
        latest / "trainer_state.pt", map_location=device, weights_only=False
    )
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    state = {key: saved[key] for key in state}
    return saved, state


def main():  # noqa: C901
    args = parse_args()
    if args.synthetic and args.max_samples is None:
        args.max_samples = 10000
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.synthetic:
        torch.set_num_threads(1)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError("Use a fresh run directory, or --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        save_train_command(str(args.output_dir / f"resume-{time.time_ns()}"))
    else:
        save_train_command(str(args.output_dir))
    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata["response_manifest_sha256"] = None
    if args.synthetic:
        metadata.update(teacher_sha="synthetic", dataset_sha="synthetic")
    metadata["training_policy"] = (
        "prepared_greedy_teacher_responses_online_teacher_features"
    )
    args.split = list(args.split)
    metadata["split"] = args.split
    args.split = args.split[0]
    teacher, processor, source_rows = load_teacher_and_rows(args, metadata)
    if args.synthetic:
        rows = source_rows
    else:
        response_manifest = json.loads(args.response_manifest.read_text())
        manifest_splits = response_manifest.get(
            "splits", [response_manifest.get("split")]
        )
        expected = {
            "teacher_sha": metadata["teacher_sha"],
            "dataset_sha": metadata["dataset_sha"],
            "splits": metadata["split"],
        }
        actual = {
            "teacher_sha": response_manifest.get("teacher_sha"),
            "dataset_sha": response_manifest.get("dataset_sha"),
            "splits": manifest_splits,
        }
        for key, value in expected.items():
            if actual[key] != value:
                raise ValueError(
                    f"Response manifest {key}={actual[key]!r} "
                    f"does not match training value {value!r}"
                )
        metadata["response_manifest_sha256"] = hashlib.sha256(
            args.response_manifest.read_bytes()
        ).hexdigest()
        prepared = load_from_disk(str(args.train_data))
        required_columns = {
            "input_ids",
            "loss_mask",
            "audio_url",
            "whisper_begin_index",
        }
        if not required_columns <= set(prepared.column_names):
            raise ValueError(
                "Prepared data must include input_ids, loss_mask, audio_url, "
                "and whisper_begin_index"
            )
        if args.max_samples is None:
            args.max_samples = len(prepared) * args.epochs
        rows = shuffled_dataset_epochs(
            prepared,
            epochs=args.epochs,
            seed=args.seed,
            buffer_size=args.shuffle_buffer,
        )
        metadata["max_samples"] = args.max_samples
    teacher.eval().requires_grad_(False)
    adapter = WhisperFeatureAdapter(teacher, args.target_layer_ids)
    latest = args.output_dir / "latest"
    draft = (
        load_whisper_draft(latest).to(teacher.device)
        if args.resume
        else build_whisper_draft(
            teacher, args.target_layer_ids, block_size=args.block_size
        )
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in draft.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
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
        "skipped": {"long_audio": 0, "short_response": 0},
        "truncated_responses": 0,
        "best_eal": -1.0,
        "last_eval": -1,
        "training_counts": {},
    }
    saved = None
    if args.resume:
        saved, state = load_resume(
            latest, metadata, teacher.device, optimizer, scheduler, state
        )
    print("Collecting fixed held-out teacher responses...", flush=True)
    samples = collect_validation(args, teacher, processor, metadata)
    metadata["validation_ids"] = [sample["id"] for sample in samples]
    atomic_write(args.output_dir / "run_config.json", json.dumps(metadata, indent=2))
    start = time.monotonic()
    with nullcontext(rows) if args.synthetic else closing(rows):
        if saved:
            # Replay the deterministic shuffled stream to recover its exact position.
            replayed = sum(1 for _ in islice(rows, state["consumed"]))
            if replayed != state["consumed"]:
                raise RuntimeError("Could not recover streaming data position")
            restore_rng(saved["rng"])
        if state["last_eval"] < 0:
            report = evaluate(args, teacher, processor, draft, samples, 0)
            state["last_eval"] = 0
            state["best_eal"] = report["eal"]
            snapshot(args, draft, optimizer, scheduler, state, metadata, "best")
            snapshot(args, draft, optimizer, scheduler, state, metadata, "latest")
            record_evaluation(args, report, metadata)
            print(
                f"eval step=0 EAL={report['eal']:.3f} MAL={report['mal']} "
                f"speedup={report['generation_speedup']}",
                flush=True,
            )
        target = args.stop_after or args.steps
        run_updates(
            args,
            teacher,
            processor,
            draft,
            adapter,
            optimizer=optimizer,
            scheduler=scheduler,
            rows=rows,
            samples=samples,
            state=state,
            metadata=metadata,
            start=start,
            target=target,
        )
        snapshot(args, draft, optimizer, scheduler, state, metadata, "latest")
    outcome = {
        **metadata,
        **state,
        "elapsed_seconds": time.monotonic() - start,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated()
        if teacher.device.type == "cuda"
        else 0,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "status": "complete" if state["step"] == args.steps else "stopped",
    }
    atomic_write(args.output_dir / "results.json", json.dumps(outcome, indent=2))
    if state["step"] < target:
        raise RuntimeError("Sample budget exhausted; latest checkpoint retained")
    print(f"Training {outcome['status']} at step {state['step']}", flush=True)


if __name__ == "__main__":
    main()
