from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.embedding_generator import local_hash_embedding, sentence_transformer_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate normalized static embeddings for Vercel.")
    parser.add_argument("--provider", choices=["hash", "minilm", "e5"], default="minilm")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--device", default=None, help="Examples: cuda, cuda:0, cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/processed/document_chunks.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/embeddings.json")
    args = parser.parse_args()

    chunks = json.loads(args.input.read_text(encoding="utf-8"))
    texts = [
        f"{chunk['projectName']} {chunk['section']} {chunk['text']} {' '.join(chunk.get('keywords', []))}"
        for chunk in chunks
    ]

    query_prefix = ""
    passage_prefix = ""
    if args.provider == "minilm":
        model_name = args.model or "sentence-transformers/all-MiniLM-L6-v2"
        matrix = sentence_transformer_embeddings(
            texts,
            model_name=model_name,
            device=args.device,
            batch_size=args.batch_size,
        )
        provider_name = "huggingface-feature-extraction"
    elif args.provider == "e5":
        model_name = args.model or "intfloat/e5-small-v2"
        query_prefix = "query: "
        passage_prefix = "passage: "
        matrix = sentence_transformer_embeddings(
            texts,
            model_name=model_name,
            device=args.device,
            batch_size=args.batch_size,
            prefix=passage_prefix,
        )
        provider_name = "huggingface-feature-extraction"
    else:
        matrix = np.stack([local_hash_embedding(text, args.dimension) for text in texts])
        provider_name = "local-hash-v1"
        model_name = "portfolio-hash-v1"

    records = [
        {"chunkId": chunk["id"], "vector": [round(float(value), 8) for value in vector]}
        for chunk, vector in zip(chunks, matrix)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")

    metadata_path = args.output.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata["embedding"] = {
        "provider": provider_name,
        "model": model_name,
        "dimension": int(matrix.shape[1]),
        "normalized": True,
        "queryPrefix": query_prefix,
        "passagePrefix": passage_prefix,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "documentEmbeddingsPrecomputed": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} embeddings ({matrix.shape[1]} dimensions) to {args.output}")
    print(f"Provider: {provider_name}; model: {model_name}; device: {args.device or 'auto'}")


if __name__ == "__main__":
    main()
