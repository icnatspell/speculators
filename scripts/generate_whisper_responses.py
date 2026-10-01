"""Generate pinned, greedy Whisper responses for native prepare-data.

Only token IDs, masks, and references to source audio are written. Whisper
hidden states remain ephemeral and are recomputed by the trainer as needed.
"""

import argparse
import hashlib
import io
import json
import shlex
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from train_whisper_dflash import (
    UnsupportedAudioError,
    audio_features,
    load_teacher_and_rows,
)

from speculators.provenance import find_repo_root, git_diff, git_sha, package_versions
from speculators.train.whisper import generate_whisper_tokens


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
        help=(
            "One or more source splits, e.g. train.clean.100 train.clean.360 "
            "train.other.500."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help=(
            "Optional maximum generated rows per split (omit to generate the "
            "full split)."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--shuffle-buffer", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        (args.max_samples is not None and args.max_samples < 1)
        or min(args.max_new_tokens, args.shuffle_buffer) < 1
        or len(args.split) != len(set(args.split))
    ):
        parser.error("Invalid sample limit, duplicate split, or generation setting")
    return args


def _load_seen(path: Path) -> dict[str, str | None]:
    seen = {}
    if path.exists():
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    row = json.loads(line)
                    seen[str(row["id"])] = row.get("source_split")
                except (json.JSONDecodeError, KeyError):
                    continue
    return seen


def _write_audio(row: dict, audio_dir: Path) -> str:
    audio = row["audio"]
    audio_bytes = audio.get("bytes")
    if not isinstance(audio_bytes, bytes):
        path = Path(audio["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"No reusable audio bytes or absolute audio file: {path}")
        return path.resolve().as_uri()

    audio_dir.mkdir(parents=True, exist_ok=True)
    row_id = str(row["id"]).replace("/", "_")
    audio_path = audio_dir / f"{row_id}.flac"
    if not audio_path.exists():
        audio_path.write_bytes(audio_bytes)
    return audio_path.resolve().as_uri()


def _rows_for_split(args, split, dataset_sha):
    from datasets import Audio, load_dataset  # noqa: PLC0415
    from pyarrow.dataset import ParquetFragmentScanOptions  # noqa: PLC0415

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=split,
        revision=dataset_sha,
        streaming=True,
        batch_size=1,
        fragment_scan_options=ParquetFragmentScanOptions(pre_buffer=False),
    ).cast_column("audio", Audio(decode=False))
    return iter(dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer))


def _duration_seconds(row):
    import soundfile as sf  # noqa: PLC0415

    audio = row["audio"]
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    return float(sf.info(source).duration)


def _summary(values):
    import statistics  # noqa: PLC0415

    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": ordered[int(0.50 * (len(ordered) - 1))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def main():  # noqa: C901
    args = parse_args()
    args.synthetic = False
    torch.manual_seed(args.seed)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    if args.output_file.exists() and not args.resume:
        raise FileExistsError("Use a new output file or pass --resume")

    args.splits = list(args.split)
    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    args.split = args.splits[0]
    teacher, processor, rows = load_teacher_and_rows(args, metadata)
    teacher.eval().requires_grad_(False)
    manifest = args.output_file.with_suffix(args.output_file.suffix + ".manifest.json")
    previous_manifest = None
    if args.resume and manifest.exists():
        previous_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        for key in (
            "teacher_sha",
            "dataset_sha",
            "splits",
            "max_new_tokens",
            "seed",
            "shuffle_buffer",
        ):
            if previous_manifest.get(key) != metadata.get(key):
                raise ValueError(f"Resume generation configuration changed: {key}")
    prompt_ids = list(processor.tokenizer.prefix_tokens)
    prompt = torch.tensor([prompt_ids], device=teacher.device)
    seen = _load_seen(args.output_file) if args.resume else {}
    rows_by_split = {}
    for seen_split in seen.values():
        if seen_split is not None:
            rows_by_split[seen_split] = rows_by_split.get(seen_split, 0) + 1
    written = 0
    skipped = previous_manifest.get("skipped_rows", 0) if previous_manifest else 0
    truncated = (
        previous_manifest.get("truncated_responses", 0) if previous_manifest else 0
    )
    generated_by_split = {}
    durations_by_split = defaultdict(list)
    token_counts_by_split = defaultdict(list)
    if args.resume and args.output_file.exists():
        with args.output_file.open(encoding="utf-8") as previous_output:
            for line in previous_output:
                try:
                    prior = json.loads(line)
                    prior_split = prior["source_split"]
                    durations_by_split[prior_split].append(
                        float(prior["audio_duration_seconds"])
                    )
                    token_counts_by_split[prior_split].append(sum(prior["loss_mask"]))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
    started = time.monotonic()

    with args.output_file.open("a", encoding="utf-8") as output:
        for split_index, split in enumerate(args.splits):
            split_rows = (
                rows
                if split_index == 0
                else _rows_for_split(args, split, metadata["dataset_sha"])
            )
            split_written = rows_by_split.get(split, 0)
            if args.max_samples is not None and split_written >= args.max_samples:
                generated_by_split[split] = split_written
                continue
            for row in split_rows:
                row_id = str(row["id"])
                if row_id in seen:
                    continue
                try:
                    duration = _duration_seconds(row)
                    audio = audio_features(row, processor, teacher.device)
                    audio_url = _write_audio(row, args.audio_dir)
                    tokens = generate_whisper_tokens(
                        teacher, audio, prompt, max_new_tokens=args.max_new_tokens
                    )
                except UnsupportedAudioError:
                    skipped += 1
                    continue

                ids = tokens[0].tolist()
                generated = ids[len(prompt_ids) :]
                if not generated:
                    skipped += 1
                    continue
                is_truncated = (
                    len(generated) == args.max_new_tokens
                    and generated[-1] != teacher.config.eos_token_id
                )
                truncated += is_truncated
                durations_by_split[split].append(duration)
                token_counts_by_split[split].append(len(generated))
                record = {
                    "id": row_id,
                    "input_ids": ids,
                    "loss_mask": [0] * len(prompt_ids) + [1] * len(generated),
                    "audio_url": audio_url,
                    "whisper_begin_index": len(prompt_ids),
                    "source_split": split,
                    "reference_text": row.get("text", ""),
                    "speaker_id": row.get("speaker_id"),
                    "chapter_id": row.get("chapter_id"),
                    "audio_duration_seconds": duration,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                seen[row_id] = split
                written += 1
                split_written += 1
                if written % 25 == 0:
                    print(
                        f"generated={written} skipped={skipped} truncated={truncated}",
                        flush=True,
                    )
                if args.max_samples is not None and split_written >= args.max_samples:
                    break
            generated_by_split[split] = split_written

    metadata.update(
        training_policy="fresh_greedy_teacher_responses",
        splits=args.splits,
        split_rows=generated_by_split,
        audio_duration_seconds={
            split: _summary(values) for split, values in durations_by_split.items()
        },
        generated_token_count={
            split: _summary(values) for split, values in token_counts_by_split.items()
        },
        prompt_token_ids=prompt_ids,
        generated_rows=sum(generated_by_split.values()),
        generated_this_invocation=written,
        skipped_rows=skipped,
        truncated_responses=truncated,
        elapsed_seconds=time.monotonic() - started,
        output_file=str(args.output_file.resolve()),
        audio_dir=str(args.audio_dir.resolve()),
        output_sha256=hashlib.sha256(args.output_file.read_bytes()).hexdigest(),
    )
    repo_root = find_repo_root(Path(__file__))
    metadata["git_sha"] = git_sha(repo_root)
    metadata["package_versions"] = package_versions()
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    command = [
        f"# timestamp_utc: {datetime.now(UTC).isoformat()}",
        f"# git_sha: {metadata['git_sha']}",
        *metadata["package_versions"],
        f"# argv: {shlex.join(sys.argv)}",
    ]
    (args.output_file.parent / "generation_command.txt").write_text(
        "\n".join(command) + "\n", encoding="utf-8"
    )
    if repo_root is not None:
        (args.output_file.parent / "speculators.patch").write_text(
            f"# repo: {repo_root} ({metadata['git_sha']})\n{git_diff(repo_root)}\n",
            encoding="utf-8",
        )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
