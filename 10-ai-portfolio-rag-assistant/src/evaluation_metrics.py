from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class QueryMetrics:
    question_id: str
    k: int
    relevant_ids: list[str]
    retrieved_ids: list[str]
    hit_rate: float
    precision: float
    recall: float
    reciprocal_rank: float
    average_precision: float
    ndcg: float

    def to_dict(self) -> dict:
        return asdict(self)


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def evaluate_ranking(
    question_id: str,
    relevant_ids: Sequence[str],
    retrieved_ids: Sequence[str],
    k: int,
) -> QueryMetrics:
    if k <= 0:
        raise ValueError("k must be positive")

    relevant = set(relevant_ids)
    ranked = unique_in_order(retrieved_ids)[:k]
    binary = [1 if item in relevant else 0 for item in ranked]
    relevant_retrieved = sum(binary)

    hit_rate = 1.0 if relevant_retrieved > 0 else 0.0
    precision = relevant_retrieved / k
    recall = relevant_retrieved / len(relevant) if relevant else 0.0

    reciprocal_rank = 0.0
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            reciprocal_rank = 1.0 / index
            break

    running_relevant = 0
    precision_sum = 0.0
    for index, rel in enumerate(binary, start=1):
        if rel:
            running_relevant += 1
            precision_sum += running_relevant / index
    average_precision = precision_sum / min(len(relevant), k) if relevant else 0.0

    ideal = [1] * min(len(relevant), k)
    ideal += [0] * max(0, k - len(ideal))
    dcg = _dcg(binary)
    idcg = _dcg(ideal)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return QueryMetrics(
        question_id=question_id,
        k=k,
        relevant_ids=list(relevant_ids),
        retrieved_ids=ranked,
        hit_rate=round(hit_rate, 6),
        precision=round(precision, 6),
        recall=round(recall, 6),
        reciprocal_rank=round(reciprocal_rank, 6),
        average_precision=round(average_precision, 6),
        ndcg=round(ndcg, 6),
    )


def summarize_query_metrics(metrics: Iterable[QueryMetrics]) -> dict[str, float]:
    rows = list(metrics)
    if not rows:
        return {
            "hit_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "map": 0.0,
            "ndcg": 0.0,
        }

    return {
        "hit_rate": round(sum(row.hit_rate for row in rows) / len(rows), 6),
        "precision": round(sum(row.precision for row in rows) / len(rows), 6),
        "recall": round(sum(row.recall for row in rows) / len(rows), 6),
        "mrr": round(sum(row.reciprocal_rank for row in rows) / len(rows), 6),
        "map": round(sum(row.average_precision for row in rows) / len(rows), 6),
        "ndcg": round(sum(row.ndcg for row in rows) / len(rows), 6),
    }
