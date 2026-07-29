from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


@dataclass(frozen=True)
class RetrievalOutput:
    rankings: dict[str, list[str]]
    index_build_ms: float
    total_query_ms: float
    mean_query_ms: float


class TfidfRetriever:
    def __init__(self, *, max_features: int | None = None) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
        )
        self.document_matrix: sparse.csr_matrix | None = None

    def fit(self, documents: list[str]) -> float:
        started = perf_counter()
        self.document_matrix = self.vectorizer.fit_transform(documents).tocsr()
        return (perf_counter() - started) * 1000.0

    def search(
        self,
        query_ids: list[str],
        queries: list[str],
        document_ids: list[str],
        *,
        top_k: int,
    ) -> RetrievalOutput:
        if self.document_matrix is None:
            raise RuntimeError("TF-IDF retriever must be fitted before search.")
        started = perf_counter()
        query_matrix = self.vectorizer.transform(queries).tocsr()
        scores = query_matrix @ self.document_matrix.T
        rankings: dict[str, list[str]] = {}

        for row_index, query_id in enumerate(query_ids):
            row = scores.getrow(row_index).toarray().ravel()
            k = min(top_k, len(row))
            top_indices = np.argpartition(-row, kth=k - 1)[:k]
            top_indices = top_indices[np.argsort(-row[top_indices])]
            rankings[query_id] = [document_ids[index] for index in top_indices]

        total_ms = (perf_counter() - started) * 1000.0
        return RetrievalOutput(
            rankings=rankings,
            index_build_ms=0.0,
            total_query_ms=total_ms,
            mean_query_ms=total_ms / max(1, len(query_ids)),
        )


class BM25Retriever:
    """Sparse Okapi BM25 implementation backed by a CountVectorizer matrix."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.vectorizer = CountVectorizer(
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self.counts: sparse.csr_matrix | None = None
        self.idf: np.ndarray | None = None
        self.document_lengths: np.ndarray | None = None
        self.average_document_length: float = 0.0

    def fit(self, documents: list[str]) -> float:
        started = perf_counter()
        counts = self.vectorizer.fit_transform(documents).tocsr().astype(np.float32)
        document_count = counts.shape[0]
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        self.idf = np.log1p(
            (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        ).astype(np.float32)
        self.document_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        self.average_document_length = float(self.document_lengths.mean())
        self.counts = counts
        return (perf_counter() - started) * 1000.0

    def _score_query(self, query: str) -> np.ndarray:
        if self.counts is None or self.idf is None or self.document_lengths is None:
            raise RuntimeError("BM25 retriever must be fitted before search.")

        query_vector = self.vectorizer.transform([query]).tocsr()
        term_indices = query_vector.indices
        if len(term_indices) == 0:
            return np.zeros(self.counts.shape[0], dtype=np.float32)

        query_counts = query_vector.data.astype(np.float32)
        term_frequency = self.counts[:, term_indices].toarray().astype(np.float32)
        length_normalizer = self.k1 * (
            1.0
            - self.b
            + self.b * self.document_lengths / max(self.average_document_length, 1e-9)
        )
        denominator = term_frequency + length_normalizer[:, None]
        term_scores = (
            self.idf[term_indices][None, :]
            * (term_frequency * (self.k1 + 1.0))
            / np.maximum(denominator, 1e-9)
            * query_counts[None, :]
        )
        return term_scores.sum(axis=1)

    def search(
        self,
        query_ids: list[str],
        queries: list[str],
        document_ids: list[str],
        *,
        top_k: int,
    ) -> RetrievalOutput:
        started = perf_counter()
        rankings: dict[str, list[str]] = {}
        for query_id, query in zip(query_ids, queries, strict=True):
            scores = self._score_query(query)
            k = min(top_k, len(scores))
            top_indices = np.argpartition(-scores, kth=k - 1)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            rankings[query_id] = [document_ids[index] for index in top_indices]

        total_ms = (perf_counter() - started) * 1000.0
        return RetrievalOutput(
            rankings=rankings,
            index_build_ms=0.0,
            total_query_ms=total_ms,
            mean_query_ms=total_ms / max(1, len(query_ids)),
        )
