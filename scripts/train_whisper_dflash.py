"""Bounded, in-memory Whisper/DFlash training experiment.

Run with --synthetic for a CPU smoke test without model/audio downloads.
Real audio streaming requires soundfile and scipy in addition to repo deps.
"""

import argparse
import hashlib
import io
import json
import resource
import time
from contextlib import closing, nullcontext
from pathlib import Path

import torch
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from speculators.data_generation.whisper import WhisperFeatureAdapter
from speculators.train.utils import save_train_command
from speculators.train.whisper import (
    build_whisper_draft,
    generate_whisper_tokens,
    load_whisper_draft,
    save_whisper_draft,
    train_whisper_step,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--teacher", default="openai/whisper-tiny.en")
    parser.add_argument("--teacher-revision", default="main")
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--dataset-config", default="all")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--split", default="train.clean.100")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--max-anchors", type=int, default=4)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.max_samples, args.max_anchors) < 1:
        parser.error("steps, max-samples and max-anchors must be positive")
    if args.block_size < 2 or args.max_new_tokens <= args.block_size:  # noqa: PLR2004
        parser.error("block-size must be >= 2 and max-new-tokens > block-size")
    if args.synthetic and args.device != "cpu":
        parser.error("The synthetic smoke test uses CPU")
    return args


def synthetic_teacher():
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
            max_target_positions=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
        )
    )


class UnsupportedAudioError(ValueError):
    """Audio duration outside this short-form training experiment."""


def audio_features(row, processor, device):
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    from scipy.signal import resample_poly  # noqa: PLC0415

    audio = row["audio"]
    source = io.BytesIO(audio["bytes"]) if audio["bytes"] else audio["path"]
    waveform, rate = sf.read(source, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if rate != 16000:  # noqa: PLR2004
        divisor = int(np.gcd(rate, 16000))
        waveform = resample_poly(waveform, 16000 // divisor, rate // divisor)
    if len(waveform) > 30 * 16000:
        raise UnsupportedAudioError("Audio clip exceeds 30 seconds")
    return processor(
        waveform, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)


def load_rows(args, split, dataset_sha):
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
    if getattr(args, "shuffle_buffer", 0):
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    return iter(dataset)


def load_teacher_and_rows(args, metadata):
    if args.synthetic:
        teacher = synthetic_teacher()
        rows = iter(range(args.max_samples))
        processor = None
    else:
        from huggingface_hub import HfApi  # noqa: PLC0415

        api = HfApi()
        model_sha = api.model_info(args.teacher, revision=args.teacher_revision).sha
        dataset_sha = api.dataset_info(args.dataset, revision=args.dataset_revision).sha
        metadata.update(teacher_sha=model_sha, dataset_sha=dataset_sha)
        processor = WhisperProcessor.from_pretrained(args.teacher, revision=model_sha)
        teacher = WhisperForConditionalGeneration.from_pretrained(
            args.teacher, revision=model_sha
        ).to(args.device)
        rows = load_rows(args, args.split, dataset_sha)
    return teacher, processor, rows


def make_teacher_sample(args, row, teacher, processor):
    if args.synthetic:
        audio = torch.randn(1, 4, 16)
        tokens = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9, 10, 2]])
        prompt_length = 2
    else:
        audio = audio_features(row, processor, args.device)
        prompt_ids = processor.tokenizer.prefix_tokens
        prompt = torch.tensor([prompt_ids], device=args.device)
        prompt_length = len(prompt_ids)
        tokens = generate_whisper_tokens(
            teacher, audio, prompt, max_new_tokens=args.max_new_tokens
        )
    return audio, tokens, prompt_length


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    # Tiny CPU matrix operations are slower with a large thread pool.
    if args.synthetic:
        torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "draft.safetensors").exists():
        raise FileExistsError("Choose a new output directory to preserve checkpoints")
    save_train_command(str(args.output_dir))
    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    teacher, processor, rows = load_teacher_and_rows(args, metadata)
    adapter = WhisperFeatureAdapter(teacher, args.target_layer_ids)
    draft = build_whisper_draft(
        teacher, adapter.target_layer_ids, block_size=args.block_size
    )
    optimizer = torch.optim.AdamW(
        [p for p in draft.parameters() if p.requires_grad], lr=1e-3
    )
    losses = []
    cuda_memory = []
    consumed = 0
    start = time.monotonic()
    # Close partially consumed Arrow scanners, including when training fails.
    with nullcontext(rows) if args.synthetic else closing(rows):
        for row in rows:
            consumed += 1
            audio, tokens, prompt_length = make_teacher_sample(
                args, row, teacher, processor
            )
            features = adapter.extract(audio, tokens, prompt_length=prompt_length)
            if features["loss_mask"][0, : -draft.block_size].any():
                losses.append(
                    train_whisper_step(
                        draft, optimizer, features, max_anchors=args.max_anchors
                    )
                )
                print(f"step={len(losses)} loss={losses[-1]:.6f}", flush=True)
            del features, audio, tokens
            if args.device != "cpu":
                cuda_memory.append(torch.cuda.memory_allocated(teacher.device))
            if len(losses) == args.steps or consumed == args.max_samples:
                break
    if len(losses) != args.steps:
        raise RuntimeError(f"Only {len(losses)} updates from {consumed} samples")
    save_whisper_draft(draft, args.output_dir)
    restored = load_whisper_draft(args.output_dir)
    for name, value in draft.state_dict().items():
        torch.testing.assert_close(value.cpu(), restored.state_dict()[name])
    metadata.update(
        losses=losses,
        consumed_samples=consumed,
        elapsed_seconds=time.monotonic() - start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()
        if args.device != "cpu"
        else 0,
        checkpoint_reload_verified=True,
        cuda_allocated_bytes_after_samples=cuda_memory,
        checkpoint_sha256=hashlib.sha256(
            (args.output_dir / "draft.safetensors").read_bytes()
        ).hexdigest(),
    )
    (args.output_dir / "results.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
