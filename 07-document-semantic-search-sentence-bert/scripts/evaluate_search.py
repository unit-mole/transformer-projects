#!/usr/bin/env python
"""Evaluate Sentence-BERT retrieval with Recall@K and MRR."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from src.model_evaluation import evaluate_queries
from src.semantic_search import load_index, search_with_vector


def main() -> None:
    processed = PROJECT_ROOT / "data/processed"
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    chunks, matrix, model_name = load_index(
        processed / "document_chunks.json", processed / "embeddings.json"
    )
    queries = json.loads((processed / "evaluation_queries.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(model_name)

    def search(query: str, top_k: int):
        vector = model.encode(query, normalize_embeddings=True)
        return search_with_vector(vector, chunks, matrix, top_k=top_k)

    metrics = evaluate_queries(queries, search)
    metrics.update({
        "status": "completed",
        "model_name": model_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "recall_at_k_results.json").write_text(json.dumps({k: v for k, v in metrics.items() if k.startswith("recall_at_") or k in {"status", "query_count"}}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "mrr_results.json").write_text(json.dumps({"status": "completed", "mrr": metrics["mrr"]}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
