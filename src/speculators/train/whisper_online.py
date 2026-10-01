"""Restart state and metrics for bounded online Whisper distillation."""

import math
import queue
import random
import shutil
import threading
from contextlib import contextmanager
from itertools import chain

import torch
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from speculators.provenance import atomic_write
from speculators.train.whisper import save_whisper_draft


def learning_rate_scale(step, *, total_steps, warmup_steps, minimum_ratio):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return minimum_ratio + (1 - minimum_ratio) * (1 + math.cos(math.pi * progress)) / 2


def shuffled_dataset_epochs(dataset, *, epochs, seed, buffer_size):
    """Globally shuffle indexed rows; avoid split-local buffered shuffle bias."""
    del buffer_size  # retained for CLI compatibility; indexed data needs no buffer
    return chain.from_iterable(
        dataset.shuffle(seed=seed + epoch) for epoch in range(epochs)
    )


class PrefetchIterator:
    """Ordered bounded producer with exclusive, quiescent pause points."""

    def __init__(self, source, function, capacity):
        if capacity < 0:
            raise ValueError("Prefetch capacity cannot be negative")
        self.source = iter(source)
        self.function = function
        self.capacity = capacity
        self.output = queue.Queue(maxsize=max(1, capacity))
        self.stopped = threading.Event()
        self.condition = threading.Condition()
        self.paused = False
        self.active = False
        self.worker = None
        self.finished = object()

    def _put(self, value):
        while not self.stopped.is_set():
            try:
                self.output.put(value, timeout=0.1)
                return
            except queue.Full:
                continue

    def _produce(self):
        try:
            for item in self.source:
                with self.condition:
                    self.condition.wait_for(
                        lambda: not self.paused or self.stopped.is_set()
                    )
                    if self.stopped.is_set():
                        return
                    self.active = True
                try:
                    value = self.function(item)
                finally:
                    with self.condition:
                        self.active = False
                        self.condition.notify_all()
                self._put(("item", value))
        except Exception as error:  # noqa: BLE001 -- relay producer errors
            self._put(("error", error))
        finally:
            self._put(self.finished)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.capacity:
            return self.function(next(self.source))
        if self.worker is None:
            self.worker = threading.Thread(
                target=self._produce, name="whisper-feature-prefetch"
            )
            self.worker.start()
        value = self.output.get()
        if value is self.finished:
            self.close()
            raise StopIteration
        kind, payload = value
        if kind == "error":
            self.close()
            raise payload
        return payload

    @contextmanager
    def pause(self):
        with self.condition:
            self.paused = True
            self.condition.wait_for(lambda: not self.active)
        try:
            yield
        finally:
            with self.condition:
                self.paused = False
                self.condition.notify_all()

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        if self.worker is not None:
            self.worker.join()
        while not self.output.empty():
            self.output.get_nowait()


def prefetch_map(source, function, *, capacity):
    return PrefetchIterator(source, function, capacity)


def normalize_transcript(text):

    return EnglishTextNormalizer({})(text)


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, observed in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != observed),
                )
            )
        previous = current
    return previous[-1]


def corpus_error_rates(references, hypotheses, *, normalizer=normalize_transcript):
    word_errors = word_count = char_errors = char_count = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        normalized_reference = normalizer(reference)
        normalized_hypothesis = normalizer(hypothesis)
        reference_words = normalized_reference.split()
        hypothesis_words = normalized_hypothesis.split()
        word_errors += edit_distance(reference_words, hypothesis_words)
        word_count += len(reference_words)
        reference_chars = normalized_reference.replace(" ", "")
        hypothesis_chars = normalized_hypothesis.replace(" ", "")
        char_errors += edit_distance(reference_chars, hypothesis_chars)
        char_count += len(reference_chars)
    return {
        "wer": word_errors / word_count if word_count else None,
        "word_errors": word_errors,
        "reference_words": word_count,
        "character_errors": char_errors,
        "reference_characters": char_count,
        "cer": char_errors / char_count if char_count else None,
    }


def capture_rng():
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(destination, draft, optimizer, scheduler, *, state, metadata):
    """Publish a complete snapshot; retain only the named latest/best snapshots."""
    import json  # noqa: PLC0415

    temporary = destination.with_name(destination.name + ".pending")
    backup = destination.with_name(destination.name + ".previous")
    if temporary.exists():
        shutil.rmtree(temporary)
    save_whisper_draft(draft, temporary)
    torch.save(
        {
            **state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng(),
        },
        temporary / "trainer_state.pt",
    )
    atomic_write(temporary / "results.json", json.dumps(metadata, indent=2))
    for filename in ("train_command.txt", "speculators.patch"):
        shutil.copy2(destination.parent / filename, temporary / filename)
    for invocation in sorted(destination.parent.glob("resume-*")):
        if invocation.is_dir():
            shutil.copytree(invocation, temporary / "provenance" / invocation.name)
    for filename in ("run_config.json",):
        if (destination.parent / filename).exists():
            shutil.copy2(destination.parent / filename, temporary / filename)
    # A checksum is part of the complete snapshot, before publication.
    from speculators.train.whisper_runtime import (  # noqa: PLC0415
        copy_directory_if_present,
        hash_file,
    )

    copy_directory_if_present(
        destination.parent / "data_provenance", temporary / "data_provenance"
    )

    atomic_write(
        temporary / "drafter_checkpoint_sha256.txt",
        hash_file(temporary / "draft.safetensors") + "\n",
    )
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    temporary.rename(destination)
    if backup.exists():
        shutil.rmtree(backup)


def acceptance_metrics(results, *, block_size):
    rounds = sum(result.draft_rounds for result in results)
    accepted = sum(result.accepted_tokens for result in results)
    proposed = sum(result.proposed_tokens for result in results)
    proposed_by_position = []
    accepted_by_position = []
    per_position = []
    for position in range(block_size - 1):
        denominator = sum(result.proposed_by_position[position] for result in results)
        numerator = sum(result.accepted_by_position[position] for result in results)
        proposed_by_position.append(denominator)
        accepted_by_position.append(numerator)
        per_position.append(numerator / denominator if denominator else None)
    eal = accepted / rounds if rounds else 0.0
    return {
        "draft_rounds": rounds,
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "acceptance_rate": accepted / proposed if proposed else None,
        "accepted_draft_length": eal,
        "eal": 1 + eal if rounds else None,
        "mal": 1 + eal if rounds else None,
        "maximum_mal": block_size,
        "proposed_by_position": proposed_by_position,
        "accepted_by_position": accepted_by_position,
        "acceptance_by_position": per_position,
        "conditional_acceptance_by_position": [
            sum(r.accepted_by_position[i] for r in results) / eligible
            if (
                eligible := sum(
                    r.eligible_by_position[i]
                    if r.eligible_by_position is not None
                    else min(
                        r.proposed_by_position[i],
                        r.draft_rounds if i == 0 else r.accepted_by_position[i - 1],
                    )
                    for r in results
                )
            )
            else None
            for i in range(block_size - 1)
        ],
        "definitions": {
            "eal": "1 + accepted draft tokens / draft rounds; includes bonus",
            "mal": "same as EAL; excludes terminal anchor-only rounds",
            "acceptance_by_position": "accepted / proposed candidates at each position",
        },
    }
