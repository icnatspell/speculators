"""Synchronized stage timing for separate diagnostic passes, never benchmarks."""

import time
from contextlib import contextmanager

import torch


class WhisperGenerationTimer:
    """Time first-token availability to last-token availability, excluding TTFT."""

    def __init__(self, device, enabled):
        self.device = device
        self.enabled = enabled
        self.start = None

    def begin(self):
        if self.enabled and self.start is None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.start = time.perf_counter()

    def finish(self):
        if not self.enabled:
            return None
        if self.start is None:
            return 0.0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - self.start


class WhisperStageTimer:
    def __init__(self, device, enabled):
        self.device = device
        self.enabled = enabled
        self.times = {}
        self.start = time.perf_counter()

    def synchronize(self):
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        self.synchronize()
        start = time.perf_counter()
        yield
        self.synchronize()
        self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - start

    def finish(self):
        if not self.enabled:
            return None
        self.synchronize()
        total = time.perf_counter() - self.start
        self.times["other"] = total - sum(self.times.values())
        self.times["total"] = total
        return self.times
