"""Isolated checkpoint benchmark on untouched LibriSpeech test splits."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from train_whisper_dflash import audio_features, load_teacher_and_rows

from speculators.provenance import atomic_write
from speculators.train.whisper import generate_whisper_tokens, load_whisper_draft
from speculators.train.whisper_eval import (
    audio_duration,
    collect_samples,
    evaluate_samples,
    load_rows,
)
from speculators.train.whisper_runtime import (
    copy_directory_if_present,
    hash_file,
    precision_dtype,
    write_provenance,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", nargs="+", default=["test.clean", "test.other"])
    parser.add_argument(
        "--samples",
        type=int,
        help="Balanced subset per split; omit for full short-form coverage",
    )
    parser.add_argument(
        "--sample-cache", type=Path, help="Reusable fixed subset mel/token cache"
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--draft-device",
        help="Defaults to verifier device; separate GPU results are labelled",
    )
    parser.add_argument(
        "--precision", choices=["auto", "float32", "bfloat16", "float16"]
    )
    parser.add_argument("--teacher-attention", choices=["eager", "sdpa"])
    parser.add_argument(
        "--allow-token-mismatch",
        action="store_true",
        help="Report reduced-precision token drift and WER instead of aborting",
    )
    parser.add_argument(
        "--cache-draft-context", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Separate synchronized stage diagnostic passes",
    )
    args = parser.parse_args()
    if args.repetitions < 1 or (args.samples is not None and args.samples < 1):
        parser.error("Counts must be positive")
    if any(not split.startswith(("test.", "validation.")) for split in args.split):
        parser.error("Benchmark must use held-out test/validation splits")
    if len(args.split) != len(set(args.split)):
        parser.error("Duplicate benchmark splits")
    return args


@torch.no_grad()
def all_short_form_samples(args, teacher, processor, coverage):
    """Stream audio and baseline targets; never retain the full test mel corpus."""
    for split in args.splits:
        coverage[split] = {
            "total_rows": 0,
            "excluded_long_audio": 0,
            "selected_rows": 0,
        }
        for row in load_rows(args, split, args.dataset_revision):
            coverage[split]["total_rows"] += 1
            duration = audio_duration(row)
            if duration > 30:  # noqa: PLR2004 -- Whisper short-form limit
                coverage[split]["excluded_long_audio"] += 1
                continue
            audio = audio_features(row, processor, teacher.device, dtype=teacher.dtype)
            prompt = torch.tensor(
                [processor.tokenizer.prefix_tokens], device=teacher.device
            )
            tokens = generate_whisper_tokens(
                teacher, audio, prompt, max_new_tokens=args.max_new_tokens
            )
            coverage[split]["selected_rows"] += 1
            yield {
                "id": row["id"],
                "split": split,
                "speaker_id": row["speaker_id"],
                "reference_text": row["text"],
                "duration": duration,
                "audio": audio.cpu(),
                "prompt": prompt.cpu(),
                "tokens": tokens.cpu(),
                "truncated": tokens[0, -1].item() != teacher.config.eos_token_id,
            }


@torch.no_grad()
def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a fresh benchmark output directory")
    args.output_dir.mkdir(parents=True)
    config_path = args.checkpoint / "run_config.json"
    if not config_path.exists():
        config_path = args.checkpoint / "results.json"
    config = json.loads(config_path.read_text())
    if config.get("synthetic"):
        raise ValueError("Use a real Whisper checkpoint for the LibriSpeech benchmark")
    run = SimpleNamespace(**config)
    run.synthetic = False
    run.skip_source_rows = True
    run.split = args.split[0]
    run.splits = args.split
    run.teacher_revision = config["teacher_sha"]
    run.dataset_revision = config["dataset_sha"]
    run.device = args.device
    run.teacher_device = args.device
    run.precision = args.precision or config.get("precision", "float32")
    run.teacher_attention = args.teacher_attention or config.get(
        "teacher_attention", "sdpa"
    )
    run.train_data = None
    teacher, processor, _ = load_teacher_and_rows(run, {})
    teacher.eval().requires_grad_(False)
    draft_device = args.draft_device or args.device
    draft = load_whisper_draft(args.checkpoint).to(draft_device).float().eval()
    draft_device = draft.embed_tokens.weight.device
    write_provenance(args.output_dir, "eval_command.txt")
    provenance = args.output_dir / "training_provenance"
    provenance.mkdir()
    for name in (
        "train_command.txt",
        "speculators.patch",
        "results.json",
        "run_config.json",
        "drafter_checkpoint_sha256.txt",
    ):
        if (args.checkpoint / name).exists():
            shutil.copy2(args.checkpoint / name, provenance / name)
    if (args.checkpoint / "provenance").exists():
        shutil.copytree(args.checkpoint / "provenance", provenance / "resumes")
    copy_directory_if_present(
        args.checkpoint / "data_provenance", provenance / "data_provenance"
    )
    coverage = {}
    if args.samples is None:
        samples = all_short_form_samples(run, teacher, processor, coverage)
    else:
        samples, coverage = collect_samples(
            run,
            teacher,
            processor,
            run.dataset_revision,
            splits=args.split,
            count=args.samples,
            cache_path=args.sample_cache or args.output_dir / "benchmark_samples.pt",
        )
    for device in {teacher.device, torch.device(draft_device)}:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
    report = evaluate_samples(
        teacher,
        processor,
        draft,
        samples,
        max_new_tokens=run.max_new_tokens,
        repetitions=args.repetitions,
        benchmark=True,
        cache_draft_context=args.cache_draft_context
        if args.cache_draft_context is not None
        else config.get("cache_draft_context", False),
        dtype=precision_dtype(run.precision, draft_device),
        profile=args.profile,
        strict_tokens=not args.allow_token_mismatch,
    )
    report.update(
        coverage=coverage,
        teacher_sha=run.teacher_revision,
        dataset_sha=run.dataset_revision,
        drafter_checkpoint_sha256=hash_file(args.checkpoint / "draft.safetensors"),
        precision=run.precision,
        teacher_attention=run.teacher_attention,
        teacher_device=str(teacher.device),
        draft_device=str(draft_device),
        teacher_dtype=str(teacher.dtype),
        multi_gpu_inference=torch.device(draft_device) != teacher.device,
        cuda_devices=[
            {"index": index, "name": torch.cuda.get_device_name(index)}
            for index in range(torch.cuda.device_count())
        ],
        peak_cuda_bytes={
            str(device): torch.cuda.max_memory_allocated(device)
            for device in {teacher.device, torch.device(draft_device)}
            if device.type == "cuda"
        },
    )
    atomic_write(args.output_dir / "results.json", json.dumps(report, indent=2) + "\n")
    (args.output_dir / "drafter_checkpoint_sha256.txt").write_text(
        report["drafter_checkpoint_sha256"] + "\n"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "mal",
                    "generation_speedup",
                    "all_tokens_match",
                    "teacher_reference_error_rates",
                    "speculative_reference_error_rates",
                    "coverage",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
