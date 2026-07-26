from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from .bi_encoder_model import BiEncoderModel
from .cross_encoder_model import CrossEncoderModel
from .dataset_loader import RankingDataset, load_ranking_dataset
from .embedding_index import NumpyEmbeddingIndex
from .reranking_pipeline import RerankingPipeline
from .retrieval_pipeline import RetrievalPipeline
from .settings import Settings


@dataclass(frozen=True)
class SearchLatency:
    index_preparation_ms: float
    query_embedding_ms: float
    retrieval_ms: float
    reranking_ms: float
    total_search_ms: float


@dataclass
class SearchResponse:
    query: str
    candidates: pd.DataFrame
    reranked_results: pd.DataFrame
    latency: SearchLatency
    models: dict[str, str]


class TwoStageRankingEngine:
    """Bi-encoder candidate retrieval followed by cross-encoder reranking."""

    def __init__(
        self,
        dataset: RankingDataset,
        settings: Settings,
        bi_encoder: BiEncoderModel | None = None,
        cross_encoder: CrossEncoderModel | None = None,
        index: NumpyEmbeddingIndex | None = None,
    ) -> None:
        self.dataset = dataset
        self.settings = settings
        self.bi_encoder = bi_encoder or BiEncoderModel(
            settings.bi_encoder_model,
            device=settings.device,
        )
        self.cross_encoder = cross_encoder or CrossEncoderModel(
            settings.cross_encoder_model,
            device=settings.device,
        )
        self.index = index or NumpyEmbeddingIndex()
        self._index_preparation_ms = 0.0

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "TwoStageRankingEngine":
        settings = settings or Settings.from_yaml()
        dataset = load_ranking_dataset(
            settings.documents_path,
            settings.queries_path,
            settings.qrels_path,
        )
        return cls(dataset=dataset, settings=settings)

    @property
    def sample_queries(self) -> list[str]:
        return self.dataset.queries["query"].astype(str).tolist()

    def _saved_index_matches(self, metadata: dict) -> bool:
        return (
            metadata.get("bi_encoder_model") == self.settings.bi_encoder_model
            and metadata.get("document_count") == len(self.dataset.documents)
            and metadata.get("document_ids")
            == self.dataset.documents["document_id"].astype(str).tolist()
        )

    def prepare_index(self, force_rebuild: bool = False, save: bool = True) -> float:
        if self.index.is_ready and not force_rebuild:
            return 0.0

        started = perf_counter()
        if not force_rebuild:
            try:
                loaded_index, metadata = NumpyEmbeddingIndex.load(
                    self.settings.index_dir
                )
                if self._saved_index_matches(metadata):
                    self.index = loaded_index
                    self._index_preparation_ms = (perf_counter() - started) * 1000
                    return self._index_preparation_ms
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                pass

        embeddings = self.bi_encoder.encode(
            self.dataset.documents["search_text"].astype(str).tolist(),
            batch_size=32,
            normalize_embeddings=True,
        )
        document_ids = self.dataset.documents["document_id"].astype(str).tolist()
        self.index.build(embeddings, document_ids)

        if save:
            self.index.save(
                self.settings.index_dir,
                metadata={
                    "bi_encoder_model": self.settings.bi_encoder_model,
                    "similarity_metric": self.settings.similarity_metric,
                    "document_ids": document_ids,
                    "dataset": "public-safe synthetic portfolio sample",
                },
            )

        self._index_preparation_ms = (perf_counter() - started) * 1000
        return self._index_preparation_ms

    def retrieve(self, query: str, candidate_k: int | None = None) -> SearchResponse:
        candidate_k = candidate_k or self.settings.default_candidate_k
        candidate_k = min(candidate_k, self.settings.maximum_candidate_k)
        index_ms = self.prepare_index()

        retrieval = RetrievalPipeline(
            self.dataset.documents,
            self.bi_encoder,
            self.index,
        ).retrieve(query, top_k=candidate_k)

        latency = SearchLatency(
            index_preparation_ms=index_ms,
            query_embedding_ms=retrieval.query_embedding_ms,
            retrieval_ms=retrieval.retrieval_ms,
            reranking_ms=0.0,
            total_search_ms=retrieval.total_ms,
        )
        return SearchResponse(
            query=query,
            candidates=retrieval.candidates,
            reranked_results=pd.DataFrame(),
            latency=latency,
            models={
                "bi_encoder": self.settings.bi_encoder_model,
                "cross_encoder": "not used",
            },
        )

    def search(
        self,
        query: str,
        candidate_k: int | None = None,
        rerank_k: int | None = None,
    ) -> SearchResponse:
        candidate_k = candidate_k or self.settings.default_candidate_k
        rerank_k = rerank_k or self.settings.default_rerank_k
        candidate_k = min(candidate_k, self.settings.maximum_candidate_k)
        rerank_k = min(rerank_k, candidate_k)

        started = perf_counter()
        index_ms = self.prepare_index()

        retrieval = RetrievalPipeline(
            self.dataset.documents,
            self.bi_encoder,
            self.index,
        ).retrieve(query, top_k=candidate_k)
        reranking = RerankingPipeline(self.cross_encoder).rerank(
            query,
            retrieval.candidates,
            rerank_k=rerank_k,
        )
        total_ms = (perf_counter() - started) * 1000

        latency = SearchLatency(
            index_preparation_ms=index_ms,
            query_embedding_ms=retrieval.query_embedding_ms,
            retrieval_ms=retrieval.retrieval_ms,
            reranking_ms=reranking.reranking_ms,
            total_search_ms=total_ms,
        )
        return SearchResponse(
            query=query,
            candidates=retrieval.candidates,
            reranked_results=reranking.results,
            latency=latency,
            models={
                "bi_encoder": self.settings.bi_encoder_model,
                "cross_encoder": self.settings.cross_encoder_model,
            },
        )
