from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TARGET = ROOT / "public/data/evaluation_summary.json"
METRICS_TARGET = OUTPUTS / "model_metrics.json"


def load_optional(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def is_measured(payload: dict) -> bool:
    return str(payload.get("status", "")).startswith("measured")


def main() -> None:
    retrieval = load_optional(OUTPUTS / "retrieval_benchmark.json")
    groundedness = load_optional(OUTPUTS / "answer_groundedness_results.json")
    citations = load_optional(OUTPUTS / "citation_correctness_results.json")
    latency = load_optional(OUTPUTS / "response_latency_results.json")
    metadata = load_optional(ROOT / "public/data/metadata.json")

    best_method = None
    best_metrics: dict = {}
    best_sort_key = (-1.0, -1.0, -1.0)
    for method in retrieval.get("methods", []):
        metrics = method.get("summary", {}).get("k=5", {})
        sort_key = (
            float(metrics.get("recall", -1)),
            float(metrics.get("ndcg", -1)),
            float(metrics.get("mrr", -1)),
        )
        if sort_key > best_sort_key:
            best_sort_key = sort_key
            best_method = method.get("method")
            best_metrics = metrics

    ground_summary = groundedness.get("summary", {}) if is_measured(groundedness) else {}
    citation_summary = citations.get("summary", {}) if is_measured(citations) else {}
    latency_summary = latency.get("summary", {}) if is_measured(latency) else {}

    measured_flags = [
        bool(retrieval.get("methods")) and retrieval.get("status") == "measured",
        is_measured(groundedness),
        is_measured(citations),
        is_measured(latency),
    ]
    status = "measured" if all(measured_flags) else ("partial" if any(measured_flags) else "pending")

    required_categories = {"ANN", "Simple RNN", "LSTM", "BiLSTM", "CNN", "Transformer"}
    categories = set(metadata.get("categories", []))
    gate_values = {
        "real_transformer_embeddings": metadata.get("embedding", {}).get("provider") == "huggingface-feature-extraction",
        "complete_category_coverage": required_categories.issubset(categories),
        "at_least_40_questions": (
            int(retrieval.get("question_count") or 0) + int(retrieval.get("unsupported_question_count") or 0)
        ) >= 40,
        "recall_at_5_at_least_0_80": float(best_metrics.get("recall", 0)) >= 0.80,
        "ndcg_at_5_at_least_0_75": float(best_metrics.get("ndcg", 0)) >= 0.75,
        "groundedness_at_least_0_85": float(ground_summary.get("mean_groundedness", 0)) >= 0.85,
        "citation_precision_at_least_0_85": float(citation_summary.get("mean_citation_precision", 0)) >= 0.85,
        "citation_completeness_at_least_0_85": float(citation_summary.get("mean_citation_completeness", 0)) >= 0.85,
        "refusal_accuracy_at_least_0_80": float(ground_summary.get("refusal_accuracy") or 0) >= 0.80,
    }

    payload = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "coverage_status": metadata.get("coverageStatus"),
            "document_count": metadata.get("documentCount"),
            "chunk_count": metadata.get("chunkCount"),
            "categories": metadata.get("categories", []),
        },
        "models": {
            "embedding": metadata.get("embedding", {}).get("model"),
            "embedding_provider": metadata.get("embedding", {}).get("provider"),
            "groundedness_evaluator": groundedness.get("nli_model") if is_measured(groundedness) else None,
        },
        "retrieval": {
            "best_method": best_method,
            "hit_rate_at_5": best_metrics.get("hit_rate"),
            "precision_at_5": best_metrics.get("precision"),
            "recall_at_5": best_metrics.get("recall"),
            "mrr_at_5": best_metrics.get("mrr"),
            "map_at_5": best_metrics.get("map"),
            "ndcg_at_5": best_metrics.get("ndcg"),
            "question_count": retrieval.get("question_count"),
            "unsupported_question_count": retrieval.get("unsupported_question_count"),
        },
        "groundedness": ground_summary,
        "citations": citation_summary,
        "latency": latency_summary,
        "quality_gates": {
            "passed": sum(gate_values.values()),
            "total": len(gate_values),
            "all_passed": all(gate_values.values()),
            "details": gate_values,
        },
        "source_files": {
            "retrieval": "outputs/retrieval_benchmark.json",
            "groundedness": "outputs/answer_groundedness_results.json",
            "citations": "outputs/citation_correctness_results.json",
            "latency": "outputs/response_latency_results.json",
        },
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    METRICS_TARGET.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {TARGET}")
    print(f"Saved {METRICS_TARGET}")


if __name__ == "__main__":
    main()
