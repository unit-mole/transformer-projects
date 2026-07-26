#!/usr/bin/env python
"""Generate and save normalized Sentence-BERT embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_generator import DEFAULT_MODEL, generate_embeddings, save_embedding_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=PROJECT_ROOT / "data/processed/document_chunks.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/embeddings.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    payload = generate_embeddings(chunks, args.model, args.batch_size)
    save_embedding_payload(payload, args.output)
    print(f"Saved {len(payload['embeddings'])} embeddings to {args.output}")


if __name__ == "__main__":
    main()
