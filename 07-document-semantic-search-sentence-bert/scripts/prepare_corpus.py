#!/usr/bin/env python
"""Load, preprocess, chunk, and summarize the document corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_chunking import chunk_documents
from src.document_loader import load_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/raw_documents")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(args.input_dir)
    if not documents:
        raise SystemExit(f"No supported documents found in {args.input_dir}")
    chunks = chunk_documents(documents, args.chunk_size, args.chunk_overlap)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus = [document.to_dict() for document in documents]
    word_counts = [len(document.text.split()) for document in documents]
    chunk_word_counts = [chunk["word_count"] for chunk in chunks]
    metadata = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "average_document_words": round(sum(word_counts) / len(word_counts), 2),
        "average_chunk_words": round(sum(chunk_word_counts) / len(chunk_word_counts), 2),
        "chunk_size_words": args.chunk_size,
        "chunk_overlap_words": args.chunk_overlap,
        "project_categories": dict(Counter(doc.project_category for doc in documents)),
        "document_types": dict(Counter(doc.document_type for doc in documents)),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "browser_model": "Xenova/all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "similarity_metric": "cosine",
        "data_safety": "Public, self-authored, redistributable, or synthetic documents only.",
    }

    (args.output_dir / "corpus.json").write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "document_chunks.json").write_text(json.dumps(chunks, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(documents)} documents and {len(chunks)} chunks in {args.output_dir}")


if __name__ == "__main__":
    main()
