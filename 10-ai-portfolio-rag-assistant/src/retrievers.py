from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.embedding_generator import local_hash_embedding


@dataclass(frozen=True)
class RankedResult:
    indices: list[int]
    scores: list[float]
    query_latency_ms: float
    retrieval_latency_ms: float


class TfidfRetriever:
    def __init__(self, documents: Sequence[str]) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int) -> RankedResult:
        start = perf_counter()
        query_vector = self.vectorizer.transform([query])
        query_ms = (perf_counter() - start) * 1000

        start = perf_counter()
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        indices = np.argsort(scores)[::-1][:top_k]
        retrieval_ms = (perf_counter() - start) * 1000
        return RankedResult(
            indices=indices.tolist(),
            scores=[float(scores[i]) for i in indices],
            query_latency_ms=query_ms,
            retrieval_latency_ms=retrieval_ms,
        )


class HashRetriever:
    def __init__(self, documents: Sequence[str], dimension: int = 384) -> None:
        self.dimension = dimension
        self.matrix = np.stack([local_hash_embedding(text, dimension) for text in documents])

    def search(self, query: str, top_k: int) -> RankedResult:
        start = perf_counter()
        query_vector = local_hash_embedding(query, self.dimension)
        query_ms = (perf_counter() - start) * 1000

        start = perf_counter()
        scores = self.matrix @ query_vector
        indices = np.argsort(scores)[::-1][:top_k]
        retrieval_ms = (perf_counter() - start) * 1000
        return RankedResult(
            indices=indices.tolist(),
            scores=[float(scores[i]) for i in indices],
            query_latency_ms=query_ms,
            retrieval_latency_ms=retrieval_ms,
        )


class DenseRetriever:
    def __init__(
        self,
        documents: Sequence[str],
        model_name: str,
        device: str | None = None,
        batch_size: int = 64,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers before using DenseRetriever") from exc

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.model = SentenceTransformer(model_name, device=device)
        passages = [f"{passage_prefix}{text}" for text in documents]
        self.matrix = self.model.encode(
            passages,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def encode_query(self, query: str) -> tuple[np.ndarray, float]:
        start = perf_counter()
        vector = self.model.encode(
            [f"{self.query_prefix}{query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0].astype(np.float32)
        return vector, (perf_counter() - start) * 1000

    def search(self, query: str, top_k: int) -> RankedResult:
        query_vector, query_ms = self.encode_query(query)
        start = perf_counter()
        scores = self.matrix @ query_vector
        indices = np.argsort(scores)[::-1][:top_k]
        retrieval_ms = (perf_counter() - start) * 1000
        return RankedResult(
            indices=indices.tolist(),
            scores=[float(scores[i]) for i in indices],
            query_latency_ms=query_ms,
            retrieval_latency_ms=retrieval_ms,
        )


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers before using CrossEncoderReranker") from exc
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        candidate_indices: Sequence[int],
        documents: Sequence[str],
        top_k: int,
    ) -> RankedResult:
        pairs = [(query, documents[index]) for index in candidate_indices]
        start = perf_counter()
        scores = np.asarray(self.model.predict(pairs, show_progress_bar=False), dtype=np.float32).reshape(-1)
        query_ms = 0.0
        order = np.argsort(scores)[::-1][:top_k]
        retrieval_ms = (perf_counter() - start) * 1000
        ranked_indices = [int(candidate_indices[i]) for i in order]
        return RankedResult(
            indices=ranked_indices,
            scores=[float(scores[i]) for i in order],
            query_latency_ms=query_ms,
            retrieval_latency_ms=retrieval_ms,
        )
