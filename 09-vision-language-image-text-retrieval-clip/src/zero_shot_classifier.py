from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ZeroShotPrediction:
    label: str
    similarity: float
    relative_score: float
    rank: int


def softmax(values: np.ndarray, temperature: float = 0.01) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64) / temperature
    array -= np.max(array)
    exp = np.exp(array)
    return exp / exp.sum()


def classify_from_embeddings(image_embedding: np.ndarray, label_embeddings: np.ndarray, labels: Sequence[str]) -> list[ZeroShotPrediction]:
    image = np.asarray(image_embedding, dtype=np.float32).reshape(-1)
    matrix = np.asarray(label_embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != image.size or matrix.shape[0] != len(labels):
        raise ValueError("labels and embedding dimensions must match")
    similarities = matrix @ image / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(image))
    probabilities = softmax(similarities)
    order = np.argsort(-similarities)
    return [
        ZeroShotPrediction(labels[index], float(similarities[index]), float(probabilities[index]), rank + 1)
        for rank, index in enumerate(order)
    ]
