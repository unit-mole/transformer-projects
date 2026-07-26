from __future__ import annotations

from statistics import mean, median

import pandas as pd

from .ranking_engine import TwoStageRankingEngine


def benchmark_latency(
    engine: TwoStageRankingEngine,
    queries: list[str],
    candidate_values: tuple[int, ...] = (5, 10, 15, 20),
    repeats: int = 3,
) -> pd.DataFrame:
    if not queries:
        raise ValueError("At least one benchmark query is required.")

    engine.search(
        queries[0],
        candidate_k=max(candidate_values),
        rerank_k=max(candidate_values),
    )

    rows: list[dict] = []
    for candidate_k in candidate_values:
        retrieval_times: list[float] = []
        reranking_times: list[float] = []
        total_times: list[float] = []

        for _ in range(repeats):
            for query in queries:
                response = engine.search(
                    query,
                    candidate_k=candidate_k,
                    rerank_k=candidate_k,
                )
                retrieval_times.append(response.latency.retrieval_ms)
                reranking_times.append(response.latency.reranking_ms)
                total_times.append(response.latency.total_search_ms)

        rows.append(
            {
                "candidate_k": candidate_k,
                "observations": len(total_times),
                "retrieval_mean_ms": mean(retrieval_times),
                "reranking_mean_ms": mean(reranking_times),
                "total_mean_ms": mean(total_times),
                "total_median_ms": median(total_times),
                "total_p95_ms": float(pd.Series(total_times).quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)
