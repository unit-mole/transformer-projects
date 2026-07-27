from __future__ import annotations

from collections.abc import Callable
from statistics import mean
from time import perf_counter
from typing import Any


def benchmark(function: Callable[..., Any], *args: Any, repeats: int = 5, **kwargs: Any) -> dict[str, float]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    timings: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        function(*args, **kwargs)
        timings.append((perf_counter() - started) * 1000)
    return {"average_ms": mean(timings), "minimum_ms": min(timings), "maximum_ms": max(timings), "repeats": repeats}
