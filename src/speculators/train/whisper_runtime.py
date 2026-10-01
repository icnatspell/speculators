"""Shared identity, precision and portable audio helpers for Whisper experiments."""

import hashlib
import json
import shlex
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch

from speculators.provenance import (
    atomic_write,
    find_repo_root,
    git_diff,
    git_sha,
    package_versions,
)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_identity(directory):
    """Content identity, including preparation configuration, independent of path."""
    directory = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.name.startswith("cache-"):
            # datasets.shuffle() writes derived index caches beside immutable rows.
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(hash_file(path).encode())
    return digest.hexdigest()


def write_provenance(directory, filename, *, argv=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = find_repo_root(Path(__file__))
    sha = git_sha(root)
    content = [
        f"# timestamp_utc: {datetime.now(UTC).isoformat()}",
        f"# git_sha: {sha}",
        *package_versions(),
        *package_versions(("datasets", "soundfile", "scipy", "safetensors")),
        f"# argv: {shlex.join(sys.argv if argv is None else argv)}",
    ]
    atomic_write(directory / filename, "\n".join(content) + "\n")
    atomic_write(directory / "speculators.patch", git_diff(root) + "\n")


def precision_dtype(precision, device):
    device = torch.device(device)
    if precision == "auto":
        if device.type != "cuda":
            return torch.float32
        with torch.cuda.device(device):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[precision]


def configure_whisper_policy(teacher, processor):
    """Declare the complete short-form English transcription prompt."""
    multilingual = getattr(teacher.generation_config, "is_multilingual", False)
    processor.tokenizer.set_prefix_tokens(
        language="english" if multilingual else None,
        task="transcribe" if multilingual else None,
        predict_timestamps=False,
    )
    teacher.generation_config.return_timestamps = False
    if multilingual:
        teacher.generation_config.language = "en"
        teacher.generation_config.task = "transcribe"


def audio_path(row, audio_root=None):
    if audio_root is not None:
        relative = row.get("audio_relative_path")
        if relative is None:
            relative = Path(unquote(urlparse(row["audio_url"]).path)).name
        path = (Path(audio_root).resolve() / relative).resolve()
        if not path.is_relative_to(Path(audio_root).resolve()):
            raise ValueError("Audio path escapes audio-root")
        return path
    parsed = urlparse(row["audio_url"])
    if parsed.scheme != "file":
        raise ValueError("Prepared Whisper audio_url must use file:// or --audio-root")
    return Path(unquote(parsed.path))


def batched_rows(rows, batch_size, *, bucket_buffer=0):
    """Bounded token-length buckets; deterministic and replayable by batch count."""
    from itertools import islice  # noqa: PLC0415

    source = iter(rows)
    window = max(batch_size, bucket_buffer)
    while items := list(islice(source, window)):
        if bucket_buffer:
            items.sort(key=lambda row: len(row["input_ids"]))
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]


def pack_whisper_features(rows):
    """Pack independent utterances; position IDs reset and document IDs isolate KV."""
    if not rows:
        raise ValueError("Cannot pack an empty feature batch")
    result = {key: torch.cat([row[key] for row in rows], dim=1) for key in rows[0]}
    result["document_ids"] = torch.cat(
        [torch.full_like(row["input_ids"], index) for index, row in enumerate(rows)],
        dim=1,
    )
    return result


def recover_checkpoint(destination):
    """Recover a complete snapshot left behind by a publication interruption."""
    destination = Path(destination)
    required = (
        "draft.safetensors",
        "whisper_draft.json",
        "trainer_state.pt",
        "results.json",
    )

    def complete(path):
        return all((path / name).is_file() for name in required)

    if complete(destination):
        return destination
    for suffix in (".previous", ".pending"):
        candidate = destination.with_name(destination.name + suffix)
        if complete(candidate):
            if destination.exists():
                raise RuntimeError(
                    f"Incomplete checkpoint at {destination}; retained {candidate}"
                )
            candidate.rename(destination)
            return destination
    raise FileNotFoundError(f"No complete checkpoint found at {destination}")


def repair_jsonl_tail(path):
    """Discard only a malformed final line; malformed interior rows are errors."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r+b") as handle:
        previous_end = 0
        while line := handle.readline():
            end = handle.tell()
            try:
                json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if handle.read(1):
                    raise ValueError("Malformed interior response row") from error
                handle.truncate(previous_end)
                return
            previous_end = end
        if previous_end:
            handle.seek(previous_end - 1)
            if handle.read(1) != b"\n":
                handle.seek(0, 2)
                handle.write(b"\n")


def copy_directory_if_present(source, destination):
    if Path(source).is_dir():
        shutil.copytree(source, destination)
