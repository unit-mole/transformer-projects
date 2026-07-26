from __future__ import annotations

from statistics import mean
from time import perf_counter
from typing import Callable, Iterable, Sequence

def benchmark(callable_: Callable[[], object], repeats: int = 5, warmup: int = 1) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    for _ in range(max(0, warmup)):
        callable_()
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        callable_()
        samples.append(perf_counter() - started)
    return {
        "repeats": repeats,
        "average_seconds": round(mean(samples), 6),
        "minimum_seconds": round(min(samples), 6),
        "maximum_seconds": round(max(samples), 6),
        "samples_seconds": [round(value, 6) for value in samples],
    }
