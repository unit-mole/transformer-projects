"""Cosine-similarity semantic search for offline evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def cosine_similarity_matrix(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    docs = np.asarray(matrix, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(docs, axis=1)
    denominator = np.maximum(query_norm * doc_norms, 1e-12)
    return (docs @ query) / denominator


def load_index(chunks_path: Path, embeddings_path: Path) -> tuple[list[dict[str, Any]], np.ndarray, str]:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    payload = json.loads(embeddings_path.read_text(encoding="utf-8"))
    if payload.get("status") != "ready" or not payload.get("embeddings"):
        raise ValueError("Embedding file is not ready. Run scripts/generate_embeddings.py first.")

    vector_by_id = {item["chunk_id"]: item["vector"] for item in payload["embeddings"]}
    missing = [chunk["chunk_id"] for chunk in chunks if chunk["chunk_id"] not in vector_by_id]
    if missing:
        raise ValueError(f"Missing embeddings for {len(missing)} chunks")
    matrix = np.asarray([vector_by_id[chunk["chunk_id"]] for chunk in chunks], dtype=np.float32)
    return chunks, matrix, str(payload["model_name"])


def search_with_vector(
    query_vector: np.ndarray,
    chunks: list[dict[str, Any]],
    matrix: np.ndarray,
    top_k: int = 5,
    category: str | None = None,
    document_type: str | None = None,
) -> list[dict[str, Any]]:
    scores = cosine_similarity_matrix(query_vector, matrix)
    candidates: list[tuple[int, float]] = []
    for index, (chunk, score) in enumerate(zip(chunks, scores, strict=True)):
        if category and chunk.get("project_category") != category:
            continue
        if document_type and chunk.get("document_type") != document_type:
            continue
        candidates.append((index, float(score)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [
        {**chunks[index], "rank": rank, "similarity_score": score}
        for rank, (index, score) in enumerate(candidates[:top_k], start=1)
    ]
