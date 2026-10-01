"""Restart state and metrics for bounded online Whisper distillation."""

import math
import queue
import random
import shutil
import threading
import unicodedata
from itertools import chain

import torch

from speculators.provenance import atomic_write
from speculators.train.whisper import save_whisper_draft


def learning_rate_scale(step, *, total_steps, warmup_steps, minimum_ratio):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return minimum_ratio + (1 - minimum_ratio) * (1 + math.cos(math.pi * progress)) / 2


def shuffled_dataset_epochs(dataset, *, epochs, seed, buffer_size):
    """Yield deterministic shuffled passes with an independent order per epoch."""
    return chain.from_iterable(
        dataset.to_iterable_dataset().shuffle(
            seed=seed + epoch, buffer_size=buffer_size
        )
        for epoch in range(epochs)
    )


def prefetch_map(source, function, *, capacity):  # noqa: C901
    """Map items in order, optionally using a bounded producer thread."""
    if capacity < 0:
        raise ValueError("Prefetch capacity cannot be negative")
    if capacity == 0:
        for item in source:
            yield function(item)
        return

    output = queue.Queue(maxsize=capacity)
    stopped = threading.Event()
    finished = object()

    def put(value):
        while not stopped.is_set():
            try:
                output.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        try:
            for item in source:
                if not put(("item", function(item))):
                    return
        except Exception as error:  # noqa: BLE001 -- relay producer errors
            put(("error", error))
        finally:
            put(finished)

    worker = threading.Thread(target=produce, name="whisper-feature-prefetch")
    worker.start()
    try:
        while True:
            value = output.get()
            if value is finished:
                break
            kind, payload = value
            if kind == "error":
                raise payload
            yield payload
    finally:
        stopped.set()
        worker.join()


def normalize_transcript(text):
    chars = [
        character.lower() if unicodedata.category(character)[0] in {"L", "N"} else " "
        for character in text
    ]
    return " ".join("".join(chars).split())


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


def corpus_error_rates(references, hypotheses):
    word_errors = word_count = char_errors = char_count = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        normalized_reference = normalize_transcript(reference)
        normalized_hypothesis = normalize_transcript(hypothesis)
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
        "definitions": {
            "eal": "1 + accepted draft tokens / draft rounds; includes bonus",
            "mal": "same as EAL; excludes terminal anchor-only rounds",
            "acceptance_by_position": "accepted / proposed candidates at each position",
        },
    }
