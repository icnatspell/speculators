"""Fixed, speaker-balanced validation and isolated generation-only benchmarks."""

import hashlib
import random
import statistics
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch

from speculators.data_generation.whisper import WhisperFeatureAdapter
from speculators.proposals.whisper import (
    greedy_whisper_decode,
    speculative_whisper_decode,
)
from speculators.train.whisper import (
    WhisperLossOptions,
    generate_whisper_tokens,
    metrics_to_cpu,
    whisper_autocast,
    whisper_draft_loss,
)
from speculators.train.whisper_online import acceptance_metrics, corpus_error_rates
from speculators.train.whisper_runtime import batched_rows


def load_rows(args, split, dataset_sha, *, shuffle=False, ids=None):
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
        **({"filters": [("id", "in", sorted(ids))]} if ids is not None else {}),
    ).cast_column("audio", Audio(decode=False))
    if shuffle:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    iterator = iter(dataset)
    try:
        yield from iterator
    finally:
        if hasattr(iterator, "close"):
            iterator.close()


def select_balanced_ids(metadata, count, seed):
    """Round-robin speakers and duration bins, with stable seeded ranking."""
    groups = defaultdict(list)
    for row in metadata:
        bucket = min(5, int(row["duration"] // 5))
        groups[(str(row["speaker_id"]), bucket)].append(row["id"])

    def rank(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    for group in groups.values():
        group.sort(key=rank)
    # Iterate speakers first so multiple duration bins do not overweight one.
    speakers = sorted({key[0] for key in groups}, key=rank)
    selected = []
    while len(selected) < count and groups:
        for speaker in speakers:
            available = [key for key in groups if key[0] == speaker]
            if not available:
                continue
            key = min(available, key=lambda key: (len(selected) + key[1]) % 6)
            selected.append(groups[key].pop())
            if not groups[key]:
                del groups[key]
            if len(selected) == count:
                break
    return selected


def audio_duration(row):
    import io  # noqa: PLC0415

    import soundfile as sf  # noqa: PLC0415

    audio = row["audio"]
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    return float(sf.info(source).duration)


@torch.no_grad()
def collect_samples(  # noqa: C901 -- selection, cache identity and streaming data
    args, teacher, processor, dataset_sha, *, splits, count, cache_path=None
):
    from train_whisper_dflash import audio_features  # noqa: PLC0415

    identity = {
        "teacher_sha": args.teacher_revision,
        "dataset_sha": dataset_sha,
        "splits": splits,
        "count": count,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "policy": "english-transcribe-no-timestamps-v1",
        "precision": getattr(args, "precision", "float32"),
        "teacher_attention": getattr(args, "teacher_attention", "sdpa"),
    }
    if cache_path and Path(cache_path).exists():
        cached = torch.load(cache_path, weights_only=True)
        if cached["identity"] != identity:
            raise ValueError("Validation cache identity changed")
        return cached["samples"], cached["coverage"]
    samples, coverage = [], {}
    for split in splits:
        candidates = []
        total = too_long = 0
        for row in load_rows(args, split, dataset_sha):
            total += 1
            duration = audio_duration(row)
            if duration > 30:  # noqa: PLR2004 -- Whisper short-form limit
                too_long += 1
                continue
            candidates.append(
                {"id": row["id"], "speaker_id": row["speaker_id"], "duration": duration}
            )
        wanted = (
            {row["id"] for row in candidates}
            if count is None
            else set(select_balanced_ids(candidates, count, args.seed))
        )
        if count is not None and len(wanted) < count:
            raise ValueError(f"Insufficient eligible samples in {split}")
        coverage[split] = {
            "total_rows": total,
            "excluded_long_audio": too_long,
            "eligible_rows": len(candidates),
            "selected_rows": len(wanted),
        }
        for row in load_rows(args, split, dataset_sha, ids=wanted):
            if row["id"] not in wanted:
                continue
            audio = audio_features(row, processor, teacher.device, dtype=teacher.dtype)
            prompt = torch.tensor(
                [processor.tokenizer.prefix_tokens], device=teacher.device
            )
            tokens = generate_whisper_tokens(
                teacher, audio, prompt, max_new_tokens=args.max_new_tokens
            )
            samples.append(
                {
                    "id": row["id"],
                    "split": split,
                    "speaker_id": row["speaker_id"],
                    "reference_text": row["text"],
                    "duration": audio_duration(row),
                    "audio": audio.cpu(),
                    "prompt": prompt.cpu(),
                    "tokens": tokens.cpu(),
                    "truncated": tokens[0, -1].item() != teacher.config.eos_token_id,
                }
            )
            wanted.remove(row["id"])
            if not wanted:
                break
    if cache_path:
        path = Path(cache_path)
        temporary = path.with_suffix(".pending")
        torch.save(
            {"identity": identity, "samples": samples, "coverage": coverage}, temporary
        )
        temporary.replace(path)
    return samples, coverage


@torch.no_grad()
def validation_loss(
    teacher,
    draft,
    samples,
    *,
    max_anchors,
    options=None,
    batch_size=1,
    dtype=torch.float32,
    seed=42,
):
    draft.eval()
    adapter = WhisperFeatureAdapter(teacher, draft.target_layer_ids)
    counts = defaultdict(float)
    by_split = {}
    for split in sorted({sample["split"] for sample in samples}):
        split_counts = defaultdict(float)
        for batch_index, batch in enumerate(
            batched_rows((s for s in samples if s["split"] == split), batch_size)
        ):
            audio = torch.cat([sample["audio"].to(teacher.device) for sample in batch])
            features = adapter.extract_batch(
                audio,
                [s["tokens"][0] for s in batch],
                prompt_lengths=[s["prompt"].shape[1] for s in batch],
            )
            for index, row in enumerate(features):
                features_row = {
                    key: value.to(draft.embed_tokens.weight.device)
                    for key, value in row.items()
                }
                # Fixed anchors without disturbing the trainer's RNG sequence.
                with torch.random.fork_rng():
                    torch.manual_seed(seed + batch_index * batch_size + index)
                    with whisper_autocast(draft.embed_tokens.weight.device, dtype):
                        _, metrics = whisper_draft_loss(
                            draft,
                            features_row,
                            max_anchors=max_anchors,
                            options=options or WhisperLossOptions(),
                            suppressed_tokens=teacher.generation_config.suppress_tokens
                            or (),
                        )
                for key, value in metrics_to_cpu(metrics).items():
                    split_counts[key] += value
        by_split[split] = summarize_loss(split_counts)
        for key, value in split_counts.items():
            counts[key] += value
    return {**summarize_loss(counts), "by_split": by_split}


def summarize_loss(counts):
    denominator = counts.get("weighted_loss_total", 0)
    return {
        "loss": counts.get("weighted_loss_sum", 0) / denominator
        if denominator
        else None,
        "loss_numerator": counts.get("weighted_loss_sum", 0),
        "loss_denominator": denominator,
        "teacher_context_eal": counts.get("eal_sum", 0) / counts["eal_total"]
        if counts.get("eal_total")
        else None,
        "metrics": dict(counts),
    }


def summarize_records(records, *, block_size, normalizer=None, bootstrap_seed=42):
    if not records:
        return {}
    base = sum(
        statistics.median(row["baseline_generation_seconds"])
        for row in records
        if row["baseline_generation_seconds"]
    )
    spec = sum(
        statistics.median(row["speculative_generation_seconds"])
        for row in records
        if row["speculative_generation_seconds"]
    )
    tokens = sum(row["generation_tokens"] for row in records)
    speculative_tokens = sum(
        row.get("speculative_generation_tokens", row["generation_tokens"])
        for row in records
    )
    rates_kwargs = {} if normalizer is None else {"normalizer": normalizer}
    references = [row["reference_text"] for row in records]
    teacher_wer = corpus_error_rates(
        references, [r["teacher_transcript"] for r in records], **rates_kwargs
    )
    speculative_wer = corpus_error_rates(
        references, [r["speculative_transcript"] for r in records], **rates_kwargs
    )
    times = [
        statistics.median(r["speculative_generation_seconds"])
        for r in records
        if r["speculative_generation_seconds"]
    ]
    result = {
        **acceptance_metrics(
            [r["decode_result"] for r in records], block_size=block_size
        ),
        "baseline_generation_seconds": base if base else None,
        "speculative_generation_seconds": spec if spec else None,
        "generation_speedup": base / spec if spec else None,
        "baseline_tokens_per_second": tokens / base if base else None,
        "speculative_tokens_per_second": speculative_tokens / spec if spec else None,
        "baseline_tpot_seconds": base / tokens if tokens and base else None,
        "speculative_tpot_seconds": spec / speculative_tokens
        if speculative_tokens and spec
        else None,
        "generation_tokens": tokens,
        "utterances": len(records),
        "generation_latency_p50_seconds": statistics.median(times) if times else None,
        "generation_latency_p95_seconds": sorted(times)[int(0.95 * (len(times) - 1))]
        if times
        else None,
        "teacher_reference_error_rates": teacher_wer,
        "speculative_reference_error_rates": speculative_wer,
        "verifier_calls": sum(r["decode_result"].verifier_calls for r in records),
    }
    if len(times) > 1:
        rng = random.Random(bootstrap_seed)
        pairs = [
            (
                statistics.median(r["baseline_generation_seconds"]),
                statistics.median(r["speculative_generation_seconds"]),
            )
            for r in records
        ]
        ratios = []
        for _ in range(500):
            drawn = rng.choices(pairs, k=len(pairs))
            denominator = sum(p[1] for p in drawn)
            if denominator:
                ratios.append(sum(p[0] for p in drawn) / denominator)
        ratios.sort()
        result["paired_bootstrap_speedup_95_percent_interval"] = (
            [
                ratios[int(0.025 * (len(ratios) - 1))],
                ratios[int(0.975 * (len(ratios) - 1))],
            ]
            if ratios
            else None
        )
    return result


@torch.no_grad()
def evaluate_samples(  # noqa: C901 -- paired benchmark and correctness gates
    teacher,
    processor,
    draft,
    samples,
    *,
    max_new_tokens,
    repetitions=1,
    benchmark=False,
    cache_draft_context=False,
    dtype=torch.float32,
    profile=False,
    strict_tokens=True,
):
    """One speculative pass for learning; isolated paired repeats for benchmarks."""
    draft.eval()
    records = []
    for sample_index, sample in enumerate(samples):
        audio = sample["audio"].to(device=teacher.device, dtype=teacher.dtype)
        prompt = sample["prompt"].to(teacher.device)
        encoder = teacher.model.encoder(audio, return_dict=True)
        shared = {
            "max_new_tokens": max_new_tokens,
            "encoder_outputs": encoder,
            "measure_generation": benchmark,
        }
        functions = {
            "baseline": partial(
                greedy_whisper_decode, teacher, audio, prompt, **shared
            ),
            "speculative": partial(
                speculative_whisper_decode,
                teacher,
                draft,
                audio,
                prompt,
                cache_draft_context=cache_draft_context,
                draft_dtype=dtype,
                **shared,
            ),
        }
        elapsed = {"baseline": [], "speculative": []}

        def check(result, sample=sample):
            matches = torch.equal(result.tokens.cpu(), sample["tokens"])
            if strict_tokens and not matches:
                raise RuntimeError(f"Token mismatch on {sample['id']}")
            return matches

        baseline_tokens = sample["tokens"]
        baseline_match = True
        speculative_match = True
        if benchmark:
            for function in functions.values():
                check(function())
        for repeat in range(repetitions if benchmark else 1):
            order = (
                ["baseline", "speculative"]
                if (repeat + sample_index) % 2 == 0
                else ["speculative", "baseline"]
            )
            for name in order if benchmark else ["speculative"]:
                result = functions[name]()
                matches = check(result)
                if name == "baseline":
                    baseline_match &= matches
                    baseline_tokens = result.tokens.cpu()
                else:
                    speculative_match &= matches
                if benchmark:
                    elapsed[name].append(result.generation_seconds)
                if name == "speculative":
                    speculative = result
        if profile:
            profiled = speculative_whisper_decode(
                teacher,
                draft,
                audio,
                prompt,
                max_new_tokens=max_new_tokens,
                encoder_outputs=encoder,
                profile=True,
                draft_dtype=dtype,
                cache_draft_context=cache_draft_context,
            )
            check(profiled)
        decode = (
            (lambda _tokens: "")
            if processor is None
            else (
                lambda tokens: processor.tokenizer.batch_decode(
                    tokens, skip_special_tokens=True
                )[0]
            )
        )
        speculative.tokens = speculative.tokens.cpu()
        records.append(
            {
                "sample_id": sample["id"],
                "split": sample["split"],
                "duration": sample.get("duration", 0),
                "reference_text": sample["reference_text"],
                "speaker_id": sample.get("speaker_id"),
                "truncated": sample.get("truncated", False),
                "teacher_transcript": decode(baseline_tokens),
                "hf_transcript": decode(sample["tokens"]),
                "speculative_transcript": decode(speculative.tokens.cpu()),
                "speculative_generation_tokens": max(
                    0, speculative.tokens.shape[1] - prompt.shape[1] - 1
                ),
                "generation_tokens": max(
                    0, baseline_tokens.shape[1] - prompt.shape[1] - 1
                ),
                "baseline_generation_seconds": elapsed["baseline"],
                "speculative_generation_seconds": elapsed["speculative"],
                "tokens_match": speculative_match,
                "baseline_matches_hf": baseline_match,
                "decode_result": speculative,
                "stage_seconds": profiled.stage_seconds if profile else None,
            }
        )
    normalizer = processor.tokenizer.normalize if processor is not None else None
    report = summarize_records(
        records, block_size=draft.block_size, normalizer=normalizer
    )
    report["by_split"] = {
        split: summarize_records(
            [r for r in records if r["split"] == split],
            block_size=draft.block_size,
            normalizer=normalizer,
        )
        for split in sorted({r["split"] for r in records})
    }
    report["duration_buckets"] = {
        f"{start}-{start + 5}s": summarize_records(
            [r for r in records if start <= r["duration"] < start + 5],
            block_size=draft.block_size,
            normalizer=normalizer,
        )
        for start in range(0, 30, 5)
    }
    report["token_length_buckets"] = {
        f"{start}-{end}": summarize_records(
            [r for r in records if start <= r["generation_tokens"] < end],
            block_size=draft.block_size,
            normalizer=normalizer,
        )
        for start, end in [(0, 32), (32, 64), (64, 128), (128, 256), (256, 449)]
    }
    for row in records:
        result = row.pop("decode_result")
        row["acceptance"] = acceptance_metrics([result], block_size=draft.block_size)
        row["verifier_calls"] = result.verifier_calls
    report.update(
        samples=records,
        all_tokens_match=all(row["tokens_match"] for row in records),
        token_match_rate=sum(row["tokens_match"] for row in records) / len(records)
        if records
        else None,
        baseline_hf_match_rate=sum(row["baseline_matches_hf"] for row in records)
        / len(records)
        if records
        else None,
        strict_tokens=strict_tokens,
        timing_scope="first_token_available_to_last_token_available",
        wer_normalizer="Whisper EnglishTextNormalizer; pinned tokenizer spelling map",
        benchmark=benchmark,
    )
    return report
