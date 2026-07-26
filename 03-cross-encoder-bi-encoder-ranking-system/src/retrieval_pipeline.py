from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from .bi_encoder_model import BiEncoderModel
from .embedding_index import NumpyEmbeddingIndex
from .text_preprocessing import clean_text


@dataclass(frozen=True)
class RetrievalResult:
    candidates: pd.DataFrame
    query_embedding_ms: float
    retrieval_ms: float

    @property
    def total_ms(self) -> float:
        return self.query_embedding_ms + self.retrieval_ms


class RetrievalPipeline:
    def __init__(
        self,
        documents: pd.DataFrame,
        bi_encoder: BiEncoderModel,
        index: NumpyEmbeddingIndex,
    ) -> None:
        self.documents = documents.reset_index(drop=True).copy()
        self.bi_encoder = bi_encoder
        self.index = index
        self._documents_by_id = self.documents.set_index("document_id", drop=False)

    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        query = clean_text(query)
        if len(query) < 3:
            raise ValueError("Enter a search query containing at least 3 characters.")

        started = perf_counter()
        query_embedding = self.bi_encoder.encode_query(query)
        query_embedding_ms = (perf_counter() - started) * 1000

        started = perf_counter()
        indices, scores = self.index.search(query_embedding, top_k=top_k)
        retrieval_ms = (perf_counter() - started) * 1000

        document_ids = [self.index.document_ids[index] for index in indices]
        candidates = self._documents_by_id.loc[document_ids].copy()
        candidates["bi_encoder_score"] = scores
        candidates["retrieval_rank"] = range(1, len(candidates) + 1)
        candidates = candidates.reset_index(drop=True)

        return RetrievalResult(
            candidates=candidates,
            query_embedding_ms=query_embedding_ms,
            retrieval_ms=retrieval_ms,
        )
