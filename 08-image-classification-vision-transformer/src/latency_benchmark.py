"""Inference latency measurement utilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class LatencyResult:
    average_ms: float
    minimum_ms: float
    maximum_ms: float
    median_ms: float
    runs: int


def benchmark(callable_: Callable[[], object], warmup: int = 5, runs: int = 30) -> LatencyResult:
    if warmup < 0 or runs <= 0:
        raise ValueError("warmup must be non-negative and runs must be positive")
    for _ in range(warmup):
        callable_()
    timings = []
    for _ in range(runs):
        start = perf_counter()
        callable_()
        timings.append((perf_counter() - start) * 1000.0)
    values = np.asarray(timings)
    return LatencyResult(float(values.mean()), float(values.min()), float(values.max()), float(np.median(values)), runs)


def as_jsonable(result: LatencyResult) -> dict:
    return asdict(result)
