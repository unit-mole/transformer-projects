"""Latency benchmarking helpers."""

from __future__ import annotations

from statistics import mean
from time import perf_counter
from typing import Callable


def benchmark_queries(
    queries: list[str],
    search_fn: Callable[[str, int], object],
    top_k_values: tuple[int, ...] = (3, 5, 10),
    repeats: int = 3,
) -> dict:
    rows = []
    for top_k in top_k_values:
        timings = []
        for _ in range(repeats):
            for query in queries:
                start = perf_counter()
                search_fn(query, top_k)
                timings.append((perf_counter() - start) * 1000)
        rows.append({
            "top_k": top_k,
            "measurements": len(timings),
            "average_ms": mean(timings),
            "minimum_ms": min(timings),
            "maximum_ms": max(timings),
        })
    return {"results": rows}
