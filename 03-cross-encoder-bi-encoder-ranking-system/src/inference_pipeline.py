from __future__ import annotations

from functools import lru_cache

from .ranking_engine import SearchResponse, TwoStageRankingEngine
from .settings import Settings


@lru_cache(maxsize=1)
def get_engine() -> TwoStageRankingEngine:
    """Create one lazy, process-level engine for local use and Gradio Spaces."""
    return TwoStageRankingEngine.from_settings(Settings.from_yaml())


def run_search(
    query: str,
    candidate_k: int = 10,
    rerank_k: int = 5,
    use_reranker: bool = True,
) -> SearchResponse:
    engine = get_engine()
    if use_reranker:
        return engine.search(
            query,
            candidate_k=candidate_k,
            rerank_k=rerank_k,
        )
    return engine.retrieve(query, candidate_k=candidate_k)
