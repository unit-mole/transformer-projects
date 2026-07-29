from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def precision_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved = ranked_ids[:k]
    return sum(relevance.get(document_id, 0) > 0 for document_id in retrieved) / k


def recall_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    relevant = {document_id for document_id, score in relevance.items() if score > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(relevant.intersection(retrieved)) / len(relevant)


def hit_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    return float(any(relevance.get(document_id, 0) > 0 for document_id in ranked_ids[:k]))


def reciprocal_rank_at_k(
    ranked_ids: list[str], relevance: dict[str, int], k: int = 10
) -> float:
    for rank, document_id in enumerate(ranked_ids[:k], start=1):
        if relevance.get(document_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    ranked_ids: list[str], relevance: dict[str, int], k: int = 100
) -> float:
    relevant = {document_id for document_id, score in relevance.items() if score > 0}
    if not relevant:
        return 0.0

    hits = 0
    accumulated_precision = 0.0
    for rank, document_id in enumerate(ranked_ids[:k], start=1):
        if document_id in relevant:
            hits += 1
            accumulated_precision += hits / rank
    return accumulated_precision / min(len(relevant), k)


def _dcg(grades: Iterable[float]) -> float:
    return sum(
        (2.0**float(grade) - 1.0) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int = 10) -> float:
    observed = [float(relevance.get(document_id, 0)) for document_id in ranked_ids[:k]]
    ideal = sorted((float(value) for value in relevance.values()), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(observed) / ideal_dcg if ideal_dcg else 0.0


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    *,
    recall_ks: tuple[int, ...] = (1, 3, 5, 10, 20, 50, 100),
    mrr_k: int = 10,
    ndcg_k: int = 10,
    map_k: int = 100,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float | str]] = []

    for query_id, relevance in qrels.items():
        ranked_ids = rankings.get(query_id, [])
        row: dict[str, float | str] = {
            "query_id": query_id,
            f"precision_at_{ndcg_k}": precision_at_k(ranked_ids, relevance, ndcg_k),
            f"hit_at_{ndcg_k}": hit_at_k(ranked_ids, relevance, ndcg_k),
            f"mrr_at_{mrr_k}": reciprocal_rank_at_k(ranked_ids, relevance, mrr_k),
            f"ndcg_at_{ndcg_k}": ndcg_at_k(ranked_ids, relevance, ndcg_k),
            f"map_at_{map_k}": average_precision_at_k(ranked_ids, relevance, map_k),
        }
        for k in recall_ks:
            row[f"recall_at_{k}"] = recall_at_k(ranked_ids, relevance, k)
        rows.append(row)

    details = pd.DataFrame(rows)
    metric_columns = [column for column in details.columns if column != "query_id"]
    summary = {
        "query_count": float(len(details)),
        **{
            column: float(pd.to_numeric(details[column], errors="coerce").mean())
            for column in metric_columns
        },
    }
    return details, summary


def paired_bootstrap_delta(
    before: np.ndarray,
    after: np.ndarray,
    *,
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 1:
        raise ValueError("Before and after arrays must be aligned one-dimensional arrays.")
    if len(before) == 0:
        return {
            "mean_before": 0.0,
            "mean_after": 0.0,
            "mean_delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "probability_delta_positive": 0.0,
        }

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(before), size=(samples, len(before)))
    deltas = (after[indices] - before[indices]).mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "mean_before": float(before.mean()),
        "mean_after": float(after.mean()),
        "mean_delta": float((after - before).mean()),
        "ci_lower": float(np.quantile(deltas, alpha / 2.0)),
        "ci_upper": float(np.quantile(deltas, 1.0 - alpha / 2.0)),
        "probability_delta_positive": float(np.mean(deltas > 0.0)),
        "bootstrap_samples": float(samples),
        "confidence": float(confidence),
    }
