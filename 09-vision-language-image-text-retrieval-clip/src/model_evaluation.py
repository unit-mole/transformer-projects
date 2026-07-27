from __future__ import annotations

from collections.abc import Iterable


def recall_at_k(ranked_ids: Iterable[Iterable[str]], relevant_ids: Iterable[set[str]], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    ranked = list(ranked_ids)
    relevant = list(relevant_ids)
    if len(ranked) != len(relevant) or not ranked:
        raise ValueError("ranked and relevant collections must be non-empty and equal length")
    hits = 0
    for predictions, truth in zip(ranked, relevant, strict=True):
        if set(list(predictions)[:k]) & set(truth):
            hits += 1
    return hits / len(ranked)


def evaluate_recall(ranked_ids: list[list[str]], relevant_ids: list[set[str]], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    return {f"recall_at_{k}": recall_at_k(ranked_ids, relevant_ids, k) for k in ks}
