from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RetrievalResult:
    image_id: str
    score: float
    rank: int


def cosine_scores(query_embedding: np.ndarray, gallery_embeddings: np.ndarray) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    gallery = np.asarray(gallery_embeddings, dtype=np.float32)
    if gallery.ndim != 2 or query.size != gallery.shape[1]:
        raise ValueError("embedding dimensions do not match")
    query_norm = np.linalg.norm(query)
    gallery_norms = np.linalg.norm(gallery, axis=1)
    if query_norm == 0 or np.any(gallery_norms == 0):
        raise ValueError("zero vector encountered")
    return (gallery @ query) / (gallery_norms * query_norm)


def rank_embeddings(query_embedding: np.ndarray, gallery_embeddings: np.ndarray, image_ids: Sequence[str], top_k: int = 5) -> list[RetrievalResult]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    scores = cosine_scores(query_embedding, gallery_embeddings)
    if len(image_ids) != len(scores):
        raise ValueError("image_ids and gallery rows must match")
    order = np.argsort(-scores)[:top_k]
    return [RetrievalResult(image_ids[index], float(scores[index]), rank + 1) for rank, index in enumerate(order)]
