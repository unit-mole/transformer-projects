from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clip_model import ClipEncoder
from src.dataset_loader import load_gallery, load_json
from src.model_evaluation import evaluate_recall
from src.retrieval_engine import rank_embeddings
from src.similarity_analysis import summarize_similarity


def main() -> None:
    web_root = PROJECT_ROOT / "web"
    gallery = load_gallery(web_root / "data" / "image_gallery.json")
    embedding_payload = load_json(web_root / "data" / "image_embeddings.json")
    if not embedding_payload.get("generated"):
        raise RuntimeError("Generate image embeddings before evaluation.")
    embedding_map = {row["image_id"]: row["embedding"] for row in embedding_payload["vectors"]}
    image_ids = [item["image_id"] for item in gallery]
    gallery_embeddings = np.asarray([embedding_map[image_id] for image_id in image_ids], dtype=np.float32)

    queries = load_json(PROJECT_ROOT / "data" / "evaluation_queries.json")["queries"]
    encoder = ClipEncoder()
    text_embeddings = encoder.encode_text([item["query"] for item in queries])
    ranked_ids: list[list[str]] = []
    relevant: list[set[str]] = []
    top_scores: list[float] = []
    for query_embedding, query in zip(text_embeddings, queries, strict=True):
        results = rank_embeddings(query_embedding, gallery_embeddings, image_ids, top_k=min(10, len(image_ids)))
        ranked_ids.append([result.image_id for result in results])
        relevant.append(set(query["relevant_image_ids"]))
        top_scores.append(results[0].score)

    metrics = evaluate_recall(ranked_ids, relevant)
    output = {
        "status": "measured",
        "model_id": "openai/clip-vit-base-patch32",
        **metrics,
        "similarity_summary": summarize_similarity(top_scores),
        "query_count": len(queries),
    }
    target = PROJECT_ROOT / "outputs" / "model_metrics.json"
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for k in (1, 5, 10):
        (PROJECT_ROOT / "outputs" / f"recall_at_{k}_results.json").write_text(
            json.dumps({"status": "measured", f"recall_at_{k}": metrics[f"recall_at_{k}"]}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
