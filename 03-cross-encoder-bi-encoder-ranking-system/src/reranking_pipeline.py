from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from .cross_encoder_model import CrossEncoderModel
from .text_preprocessing import clean_text


@dataclass(frozen=True)
class RerankingResult:
    results: pd.DataFrame
    reranking_ms: float


class RerankingPipeline:
    def __init__(self, cross_encoder: CrossEncoderModel) -> None:
        self.cross_encoder = cross_encoder

    def rerank(
        self,
        query: str,
        candidates: pd.DataFrame,
        rerank_k: int,
    ) -> RerankingResult:
        if candidates.empty:
            return RerankingResult(candidates.copy(), 0.0)

        query = clean_text(query)
        rerank_count = min(max(1, int(rerank_k)), len(candidates))
        head = candidates.head(rerank_count).copy()
        tail = candidates.iloc[rerank_count:].copy()

        started = perf_counter()
        cross_scores = self.cross_encoder.score(
            query,
            head["search_text"].astype(str).tolist(),
        )
        reranking_ms = (perf_counter() - started) * 1000

        head["cross_encoder_score"] = cross_scores
        head = head.sort_values(
            by="cross_encoder_score",
            ascending=False,
            kind="mergesort",
        ).reset_index(drop=True)
        head["reranked_rank"] = range(1, len(head) + 1)
        head["rank_movement"] = head["retrieval_rank"] - head["reranked_rank"]
        head["reranked"] = True

        if not tail.empty:
            tail["cross_encoder_score"] = pd.NA
            tail["reranked_rank"] = range(len(head) + 1, len(candidates) + 1)
            tail["rank_movement"] = 0
            tail["reranked"] = False

        combined = pd.concat([head, tail], ignore_index=True)
        return RerankingResult(combined, reranking_ms)
