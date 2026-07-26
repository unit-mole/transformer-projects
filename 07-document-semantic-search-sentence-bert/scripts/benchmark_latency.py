#!/usr/bin/env python
"""Benchmark query embedding plus ranking latency."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from src.latency_benchmark import benchmark_queries
from src.semantic_search import load_index, search_with_vector


def main() -> None:
    processed = PROJECT_ROOT / "data/processed"
    output_path = PROJECT_ROOT / "outputs/query_latency_results.json"
    chunks, matrix, model_name = load_index(
        processed / "document_chunks.json", processed / "embeddings.json"
    )
    evaluation_queries = json.loads((processed / "evaluation_queries.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(model_name)

    def search(query: str, top_k: int):
        vector = model.encode(query, normalize_embeddings=True)
        return search_with_vector(vector, chunks, matrix, top_k=top_k)

    report = benchmark_queries([item["query"] for item in evaluation_queries], search)
    report.update({
        "status": "completed",
        "model_name": model_name,
        "corpus_size_chunks": len(chunks),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Python end-to-end latency. Browser UI reports browser embedding and ranking latency separately.",
    })
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
