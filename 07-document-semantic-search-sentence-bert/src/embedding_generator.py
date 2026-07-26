"""Generate normalized Sentence-BERT document embeddings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def generate_embeddings(
    chunks: list[dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required. Run: pip install -r requirements.txt"
        ) from exc

    if not chunks:
        raise ValueError("No document chunks were provided")

    model = SentenceTransformer(model_name)
    texts = [str(chunk["text"]) for chunk in chunks]
    matrix = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    return {
        "schema_version": "1.0",
        "status": "ready",
        "strategy": "offline_precomputed",
        "model_name": model_name,
        "browser_model_name": "Xenova/all-MiniLM-L6-v2",
        "embedding_dimension": int(matrix.shape[1]),
        "normalized": True,
        "similarity_metric": "cosine",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embeddings": [
            {"chunk_id": chunk["chunk_id"], "vector": vector.tolist()}
            for chunk, vector in zip(chunks, matrix, strict=True)
        ],
    }


def save_embedding_payload(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
