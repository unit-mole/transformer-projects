from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .ranking_engine import TwoStageRankingEngine


def recall_at_k(
    ranked_document_ids: Iterable[str],
    relevance: dict[str, float],
    k: int,
) -> float:
    relevant = {doc_id for doc_id, grade in relevance.items() if grade > 0}
    if not relevant:
        return 0.0
    retrieved = set(list(ranked_document_ids)[:k])
    return len(retrieved.intersection(relevant)) / len(relevant)


def reciprocal_rank_at_k(
    ranked_document_ids: Iterable[str],
    relevance: dict[str, float],
    k: int = 10,
) -> float:
    for rank, document_id in enumerate(list(ranked_document_ids)[:k], start=1):
        if relevance.get(document_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_document_ids: Iterable[str],
    relevance: dict[str, float],
    k: int = 10,
) -> float:
    ranked = list(ranked_document_ids)[:k]

    def dcg(grades: list[float]) -> float:
        return sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(grades, start=1)
        )

    observed = [float(relevance.get(document_id, 0.0)) for document_id in ranked]
    ideal = sorted((float(value) for value in relevance.values()), reverse=True)[:k]
    denominator = dcg(ideal)
    return dcg(observed) / denominator if denominator else 0.0


def evaluate_engine(
    engine: TwoStageRankingEngine,
    split: str | None = None,
    candidate_k: int = 10,
) -> tuple[pd.DataFrame, dict]:
    queries = engine.dataset.queries
    if split and "split" in queries.columns:
        queries = queries[queries["split"] == split]

    relevance_lookup = engine.dataset.relevance_lookup
    rows: list[dict] = []

    # Warm up once so model download and initial index construction do not
    # distort per-query retrieval/reranking latency.
    if not queries.empty:
        engine.search(
            str(queries.iloc[0]["query"]),
            candidate_k=candidate_k,
            rerank_k=candidate_k,
        )

    for row in queries.itertuples(index=False):
        query_id = str(row.query_id)
        query = str(row.query)
        relevance = relevance_lookup.get(query_id, {})
        response = engine.search(
            query,
            candidate_k=candidate_k,
            rerank_k=candidate_k,
        )

        bi_ids = response.candidates["document_id"].astype(str).tolist()
        reranked_ids = (
            response.reranked_results["document_id"].astype(str).tolist()
        )

        bi_mrr = reciprocal_rank_at_k(bi_ids, relevance, 10)
        reranked_mrr = reciprocal_rank_at_k(reranked_ids, relevance, 10)
        bi_ndcg = ndcg_at_k(bi_ids, relevance, 10)
        reranked_ndcg = ndcg_at_k(reranked_ids, relevance, 10)

        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "recall_at_5": recall_at_k(bi_ids, relevance, 5),
                "recall_at_10": recall_at_k(bi_ids, relevance, 10),
                "bi_encoder_mrr_at_10": bi_mrr,
                "reranked_mrr_at_10": reranked_mrr,
                "mrr_improvement": reranked_mrr - bi_mrr,
                "bi_encoder_ndcg_at_10": bi_ndcg,
                "reranked_ndcg_at_10": reranked_ndcg,
                "ndcg_improvement": reranked_ndcg - bi_ndcg,
                "query_embedding_ms": response.latency.query_embedding_ms,
                "retrieval_ms": response.latency.retrieval_ms,
                "reranking_ms": response.latency.reranking_ms,
                "total_search_ms": response.latency.total_search_ms,
                "bi_top_document": bi_ids[0] if bi_ids else "",
                "reranked_top_document": reranked_ids[0] if reranked_ids else "",
            }
        )

    details = pd.DataFrame(rows)
    if details.empty:
        summary = {"status": "no_queries_evaluated"}
        return details, summary

    summary = {
        "status": "completed",
        "query_count": int(len(details)),
        "candidate_k": int(candidate_k),
        "recall_at_5": float(details["recall_at_5"].mean()),
        "recall_at_10": float(details["recall_at_10"].mean()),
        "bi_encoder_mrr_at_10": float(details["bi_encoder_mrr_at_10"].mean()),
        "reranked_mrr_at_10": float(details["reranked_mrr_at_10"].mean()),
        "mrr_improvement": float(details["mrr_improvement"].mean()),
        "bi_encoder_ndcg_at_10": float(
            details["bi_encoder_ndcg_at_10"].mean()
        ),
        "reranked_ndcg_at_10": float(
            details["reranked_ndcg_at_10"].mean()
        ),
        "ndcg_improvement": float(details["ndcg_improvement"].mean()),
        "average_query_embedding_ms": float(
            details["query_embedding_ms"].mean()
        ),
        "average_retrieval_ms": float(details["retrieval_ms"].mean()),
        "average_reranking_ms": float(details["reranking_ms"].mean()),
        "average_total_search_ms": float(details["total_search_ms"].mean()),
        "models": {
            "bi_encoder": engine.settings.bi_encoder_model,
            "cross_encoder": engine.settings.cross_encoder_model,
        },
        "dataset": "public-safe synthetic portfolio sample",
    }
    return details, summary
