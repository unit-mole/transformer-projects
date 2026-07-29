from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch


def resolve_device(requested: str = "auto") -> str:
    requested = requested.strip().lower()
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class DenseRetrievalOutput:
    rankings: dict[str, list[str]]
    scores: dict[str, list[float]]
    corpus_embedding_ms: float
    query_embedding_ms: float
    search_ms: float
    mean_query_ms: float
    device: str


@dataclass(frozen=True)
class RerankingOutput:
    rankings: dict[str, list[str]]
    scores: dict[str, list[float]]
    reranking_ms: float
    mean_query_ms: float
    pair_count: int
    device: str


class DenseTransformerRetriever:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 128,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)

    def retrieve(
        self,
        query_ids: list[str],
        queries: list[str],
        document_ids: list[str],
        documents: list[str],
        *,
        top_k: int,
    ) -> DenseRetrievalOutput:
        synchronize(self.device)
        started = perf_counter()
        corpus_embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        synchronize(self.device)
        corpus_embedding_ms = (perf_counter() - started) * 1000.0

        started = perf_counter()
        query_embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        synchronize(self.device)
        query_embedding_ms = (perf_counter() - started) * 1000.0

        started = perf_counter()
        similarity = query_embeddings @ corpus_embeddings.T
        rankings: dict[str, list[str]] = {}
        ranking_scores: dict[str, list[float]] = {}
        for row_index, query_id in enumerate(query_ids):
            row = similarity[row_index]
            k = min(top_k, len(row))
            top_indices = np.argpartition(-row, kth=k - 1)[:k]
            top_indices = top_indices[np.argsort(-row[top_indices])]
            rankings[query_id] = [document_ids[index] for index in top_indices]
            ranking_scores[query_id] = [float(row[index]) for index in top_indices]
        search_ms = (perf_counter() - started) * 1000.0

        return DenseRetrievalOutput(
            rankings=rankings,
            scores=ranking_scores,
            corpus_embedding_ms=corpus_embedding_ms,
            query_embedding_ms=query_embedding_ms,
            search_ms=search_ms,
            mean_query_ms=(query_embedding_ms + search_ms) / max(1, len(query_ids)),
            device=self.device,
        )


class TransformerCrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 512,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.model_name = model_name
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            max_length=max_length,
        )

    def rerank(
        self,
        query_ids: list[str],
        queries: dict[str, str],
        candidates: dict[str, list[str]],
        corpus: dict[str, dict[str, str]],
        *,
        rerank_k: int,
    ) -> RerankingOutput:
        from .beir_loader import combine_title_and_text

        pairs: list[tuple[str, str]] = []
        pair_metadata: list[tuple[str, str]] = []
        for query_id in query_ids:
            query = queries[query_id]
            for document_id in candidates.get(query_id, [])[:rerank_k]:
                document = corpus[document_id]
                pairs.append(
                    (
                        query,
                        combine_title_and_text(
                            document.get("title", ""), document.get("text", "")
                        ),
                    )
                )
                pair_metadata.append((query_id, document_id))

        synchronize(self.device)
        started = perf_counter()
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        synchronize(self.device)
        reranking_ms = (perf_counter() - started) * 1000.0
        raw_scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)

        grouped: dict[str, list[tuple[str, float]]] = {query_id: [] for query_id in query_ids}
        for (query_id, document_id), score in zip(pair_metadata, raw_scores, strict=True):
            grouped[query_id].append((document_id, float(score)))

        rankings: dict[str, list[str]] = {}
        ranking_scores: dict[str, list[float]] = {}
        for query_id in query_ids:
            ordered = sorted(grouped[query_id], key=lambda item: item[1], reverse=True)
            rankings[query_id] = [document_id for document_id, _ in ordered]
            ranking_scores[query_id] = [score for _, score in ordered]

        return RerankingOutput(
            rankings=rankings,
            scores=ranking_scores,
            reranking_ms=reranking_ms,
            mean_query_ms=reranking_ms / max(1, len(query_ids)),
            pair_count=len(pairs),
            device=self.device,
        )
