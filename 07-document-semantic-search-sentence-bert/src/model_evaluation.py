"""Retrieval metrics for semantic search."""

from __future__ import annotations

from statistics import mean
from typing import Any, Callable


def _is_relevant(result: dict[str, Any], query: dict[str, Any]) -> bool:
    relevant_documents = set(query.get("relevant_document_ids", []))
    relevant_chunks = set(query.get("relevant_chunk_ids", []))
    return (
        result.get("document_id") in relevant_documents
        or result.get("chunk_id") in relevant_chunks
    )


def recall_at_k(results: list[dict[str, Any]], query: dict[str, Any], k: int) -> float:
    return float(any(_is_relevant(item, query) for item in results[:k]))


def reciprocal_rank(results: list[dict[str, Any]], query: dict[str, Any]) -> float:
    for rank, item in enumerate(results, start=1):
        if _is_relevant(item, query):
            return 1.0 / rank
    return 0.0


def evaluate_queries(
    queries: list[dict[str, Any]],
    search_fn: Callable[[str, int], list[dict[str, Any]]],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    for query in queries:
        max_k = max(ks)
        results = search_fn(query["query"], max_k)
        row = {
            "query": query["query"],
            "reciprocal_rank": reciprocal_rank(results, query),
            "top_document_ids": [item.get("document_id") for item in results],
        }
        row.update({f"recall_at_{k}": recall_at_k(results, query, k) for k in ks})
        per_query.append(row)

    return {
        "query_count": len(queries),
        "mrr": mean(item["reciprocal_rank"] for item in per_query) if per_query else 0.0,
        **{
            f"recall_at_{k}": mean(item[f"recall_at_{k}"] for item in per_query) if per_query else 0.0
            for k in ks
        },
        "per_query": per_query,
    }
