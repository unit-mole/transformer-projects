from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np

@dataclass(frozen=True)
class RecallResult:
    question_id: str
    k: int
    expected_ids: list[str]
    retrieved_ids: list[str]
    hit: bool

    def to_dict(self) -> dict:
        return asdict(self)


def rank(query_vector: np.ndarray, matrix: np.ndarray, project_ids: list[str], k: int) -> list[str]:
    if matrix.ndim != 2 or matrix.shape[1] != query_vector.shape[0]:
        raise ValueError("Embedding dimensions do not match.")
    scores = matrix @ query_vector
    indices = np.argsort(scores)[::-1][:k]
    return [project_ids[index] for index in indices]


def recall_at_k(results: Iterable[RecallResult]) -> float:
    items = list(results)
    return 0.0 if not items else sum(item.hit for item in items) / len(items)
