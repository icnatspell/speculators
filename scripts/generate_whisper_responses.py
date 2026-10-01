"""Batched, resumable pinned teacher responses; hidden states stay ephemeral."""

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from train_whisper_dflash import (
    UnsupportedAudioError,
    audio_features,
    load_teacher_and_rows,
)

from speculators.provenance import atomic_write
from speculators.train.whisper import generate_whisper_tokens
from speculators.train.whisper_eval import audio_duration, load_rows
from speculators.train.whisper_runtime import (
    hash_file,
    repair_jsonl_tail,
    write_provenance,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="openai/whisper-tiny.en")
    parser.add_argument("--teacher-revision", default="main")
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--dataset-config", default="all")
    parser.add_argument(
        "--dataset-revision", default="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
    )
    parser.add_argument("--split", nargs="+", default=["train.clean.100"])
    parser.add_argument("--max-samples", type=int, help="Maximum output rows per split")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--shuffle-buffer", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="float32",
    )
    parser.add_argument(
        "--teacher-attention", choices=["eager", "sdpa"], default="sdpa"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-workers", type=int, default=1)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.max_new_tokens,
            args.shuffle_buffer,
            args.batch_size,
            args.audio_workers,
        )
        < 1
    ):
        parser.error("Counts must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("max-samples must be positive")
    if len(args.split) != len(set(args.split)):
        parser.error("Duplicate splits")
    return args


def _load_seen(path):
    if not Path(path).exists():
        return {}
    with Path(path).open() as source:
        return {
            str(row["id"]): row.get("source_split")
            for line in source
            if (row := json.loads(line))
        }


def _write_audio(row, audio_dir):
    audio = row["audio"]
    if not isinstance(audio.get("bytes"), bytes):
        path = Path(audio["path"]).resolve()
        if not path.is_file():
            raise ValueError(f"No reusable audio file: {path}")
        # Copy into the relocatable corpus root instead of referring elsewhere.
        data = path.read_bytes()
    else:
        data = audio["bytes"]
    audio_dir.mkdir(parents=True, exist_ok=True)
    path = audio_dir / (str(row["id"]).replace("/", "_") + ".flac")
    if not path.exists():
        temporary = path.with_suffix(".pending")
        temporary.write_bytes(data)
        temporary.replace(path)
    return path.resolve().as_uri()


def _summary(values):
    import statistics  # noqa: PLC0415

    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": ordered[int(0.5 * (len(values) - 1))],
        "p90": ordered[int(0.9 * (len(values) - 1))],
        "max": ordered[-1],
    }


def read_audio(row, processor):
    try:
        duration = audio_duration(row)
        return row, audio_features(row, processor, "cpu"), duration
    except UnsupportedAudioError:
        return row, None, None


@torch.no_grad()
def main():  # noqa: C901
    args = parse_args()
    args.synthetic = False
    args.skip_source_rows = True
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_file.with_suffix(
        args.output_file.suffix + ".manifest.json"
    )
    if args.output_file.exists() and not args.resume:
        raise FileExistsError("Use a new response file or --resume")
    if args.resume and args.output_file.exists() and not manifest_path.exists():
        raise ValueError("Cannot resume responses without an initial identity manifest")
    if args.resume:
        repair_jsonl_tail(args.output_file)
    splits = list(args.split)
    args.split = splits[0]
    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    teacher, processor, _ = load_teacher_and_rows(args, metadata)
    teacher.eval().requires_grad_(False)
    prompt_ids = list(processor.tokenizer.prefix_tokens)
    if len(prompt_ids) + args.max_new_tokens > teacher.config.max_target_positions:
        raise ValueError("Generation budget exceeds the teacher decoder position limit")
    identity = {
        "teacher_sha": metadata["teacher_sha"],
        "dataset_sha": metadata["dataset_sha"],
        "splits": splits,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "source_order": "buffer-shuffle-v1",
        "prompt_token_ids": prompt_ids,
        "precision": args.precision,
        "teacher_attention": args.teacher_attention,
        "batch_size": args.batch_size,
    }
    previous = (
        json.loads(manifest_path.read_text())
        if args.resume and manifest_path.exists()
        else {}
    )
    if previous and any(previous.get(key) != value for key, value in identity.items()):
        raise ValueError("Resume generation identity changed")
    if (
        previous.get("status") == "complete"
        and previous.get("max_samples") == args.max_samples
    ):
        if hash_file(args.output_file) != previous["output_sha256"]:
            raise ValueError("Completed response file content changed")
        print(
            "Response generation already complete; preserving its manifest", flush=True
        )
        return
    seen = _load_seen(args.output_file) if args.resume else {}
    counts = Counter(split for split in seen.values() if split is not None)
    skipped = previous.get("skipped_rows", 0)
    stats = defaultdict(list)
    durations = defaultdict(list)
    speakers = defaultdict(set)
    truncated = 0
    if args.resume and args.output_file.exists():
        with args.output_file.open() as handle:
            for line in handle:
                row = json.loads(line)
                stats[row["source_split"]].append(sum(row["loss_mask"]))
                durations[row["source_split"]].append(row["audio_duration_seconds"])
                speakers[row["source_split"]].add(row.get("speaker_id"))
                truncated += row["input_ids"][-1] != teacher.config.eos_token_id
    provenance_dir = (
        args.output_file.parent / f"resume-{time.time_ns()}"
        if previous
        else args.output_file.parent
    )
    write_provenance(provenance_dir, "generation_command.txt")
    metadata.update(
        identity,
        status="running",
        training_policy="fresh_greedy_teacher_responses",
        output_file=str(args.output_file.resolve()),
        audio_dir=str(args.audio_dir.resolve()),
    )
    # Publish immutable identity before any response row, including the first run.
    atomic_write(manifest_path, json.dumps(metadata, indent=2))
    started = time.monotonic()
    written = 0
    with (
        ThreadPoolExecutor(max_workers=args.audio_workers) as pool,
        args.output_file.open("a") as output,
    ):
        for split in splits:
            pending = []

            def flush(pending=pending, split=split):
                nonlocal written, truncated, skipped
                if not pending:
                    return
                audio = torch.cat([item[1] for item in pending]).to(
                    teacher.device, dtype=teacher.dtype
                )
                prompt = torch.tensor([prompt_ids], device=teacher.device).expand(
                    len(pending), -1
                )
                lengths = torch.tensor(
                    [item[2] for item in pending], device=audio.device
                ) * (
                    processor.feature_extractor.sampling_rate
                    / processor.feature_extractor.hop_length
                )
                attention_mask = (
                    torch.arange(audio.shape[-1], device=audio.device)[None]
                    < lengths.ceil()[:, None]
                ).long()
                sequences = generate_whisper_tokens(
                    teacher,
                    audio,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    attention_mask=attention_mask,
                ).cpu()
                for (row, _audio, duration), sequence in zip(
                    pending, sequences, strict=True
                ):
                    ids = sequence.tolist()
                    tail = ids[len(prompt_ids) :]
                    if teacher.config.eos_token_id in tail:
                        ids = ids[
                            : len(prompt_ids)
                            + tail.index(teacher.config.eos_token_id)
                            + 1
                        ]
                    uri = _write_audio(row, args.audio_dir)
                    count = len(ids) - len(prompt_ids)
                    record = {
                        "id": row["id"],
                        "input_ids": ids,
                        "loss_mask": [0] * len(prompt_ids) + [1] * count,
                        "audio_url": uri,
                        "audio_relative_path": Path(uri).name,
                        "whisper_begin_index": len(prompt_ids),
                        "source_split": split,
                        "reference_text": row.get("text", ""),
                        "speaker_id": row.get("speaker_id"),
                        "chapter_id": row.get("chapter_id"),
                        "audio_duration_seconds": duration,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    seen[str(row["id"])] = split
                    counts[split] += 1
                    written += 1
                    truncated += ids[-1] != teacher.config.eos_token_id
                    stats[split].append(count)
                    durations[split].append(duration)
                    speakers[split].add(row.get("speaker_id"))
                output.flush()
                pending.clear()
                metadata.update(
                    generated_rows=sum(counts.values()),
                    split_rows=dict(counts),
                    skipped_rows=skipped,  # noqa: B023 -- synchronous flush reads live counter
                )
                atomic_write(manifest_path, json.dumps(metadata, indent=2))
                print(f"generated={written} skipped={skipped}", flush=True)  # noqa: B023 -- synchronous flush

            source = (
                row
                for row in load_rows(args, split, metadata["dataset_sha"], shuffle=True)
                if str(row["id"]) not in seen
            )
            # executor.map without a buffersize eagerly submits an entire stream;
            # submit bounded windows explicitly to keep corpus memory bounded.
            from itertools import islice  # noqa: PLC0415

            while window := list(islice(source, args.batch_size)):
                if args.max_samples is not None and counts[split] >= args.max_samples:
                    break
                for item in pool.map(lambda row: read_audio(row, processor), window):
                    if item[1] is None:
                        skipped += 1
                        continue
                    if (
                        args.max_samples is None
                        or counts[split] + len(pending) < args.max_samples
                    ):
                        pending.append(item)
                flush()
            flush()
    metadata.update(
        status="complete",
        split_rows=dict(counts),
        generated_rows=sum(counts.values()),
        generated_this_invocation=written,
        skipped_rows=skipped,
        truncated_responses=truncated,
        elapsed_seconds=time.monotonic() - started,
        output_sha256=hash_file(args.output_file),
        audio_duration_seconds={s: _summary(v) for s, v in durations.items()},
        generated_token_count={s: _summary(v) for s, v in stats.items()},
        speakers_by_split={s: len(v) for s, v in speakers.items()},
    )
    atomic_write(manifest_path, json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
