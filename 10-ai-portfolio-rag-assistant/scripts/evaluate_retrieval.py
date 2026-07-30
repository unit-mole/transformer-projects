from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.embedding_generator import local_hash_embedding, minilm_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Retrieval Recall@K on the current corpus.")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/retrieval_recall_at_k.json")
    args = parser.parse_args()

    chunks = json.loads((PROJECT_ROOT / "public/data/document_chunks.json").read_text(encoding="utf-8"))
    records = json.loads((PROJECT_ROOT / "public/data/embeddings.json").read_text(encoding="utf-8"))
    metadata = json.loads((PROJECT_ROOT / "public/data/metadata.json").read_text(encoding="utf-8"))
    questions = json.loads((PROJECT_ROOT / "data/processed/evaluation_questions.json").read_text(encoding="utf-8"))
    matrix = np.array([record["vector"] for record in records], dtype=np.float32)

    if metadata["embedding"]["provider"] != "local-hash-v1":
        raise RuntimeError("For MiniLM evaluation, extend this script to encode questions with the same MiniLM model in the current environment.")

    details = []
    summary = {}
    for k in args.k:
        hits = 0
        for item in questions:
            query = local_hash_embedding(item["question"], metadata["embedding"]["dimension"])
            scores = matrix @ query
            indices = np.argsort(scores)[::-1][:k]
            retrieved = [chunks[index]["projectId"] for index in indices]
            expected = item["expected_source_project_ids"]
            hit = any(project_id in expected for project_id in retrieved)
            hits += int(hit)
            details.append({"questionId": item["id"], "k": k, "expected": expected, "retrieved": retrieved, "hit": hit})
        summary[f"recall@{k}"] = round(hits / len(questions), 4)

    payload = {
        "status": "measured_on_starter_corpus",
        "embeddingProvider": metadata["embedding"]["provider"],
        "questionCount": len(questions),
        "summary": summary,
        "details": details,
        "warning": "These results apply only to the bundled starter corpus and must be rerun after the full portfolio is indexed."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
