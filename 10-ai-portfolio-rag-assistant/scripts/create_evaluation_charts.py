from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def load_json(name: str) -> dict:
    path = OUTPUTS / name
    if not path.exists():
        raise FileNotFoundError(f"Run the evaluation pipeline first; missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_chart() -> None:
    payload = load_json("retrieval_benchmark.json")
    rows = []
    for method in payload.get("methods", []):
        metrics = method.get("summary", {}).get("k=5", {})
        rows.append({"Method": method.get("method"), **metrics})
    if not rows:
        return
    frame = pd.DataFrame(rows).set_index("Method")
    columns = [column for column in ["hit_rate", "precision", "recall", "mrr", "ndcg"] if column in frame]
    ax = frame[columns].plot(kind="bar", figsize=(12, 7))
    ax.set_title("Retrieval Evaluation at K=5")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUTS / "retrieval_method_comparison.png", dpi=180)
    plt.close(ax.figure)


def answer_quality_chart() -> None:
    groundedness = load_json("answer_groundedness_results.json")
    citations = load_json("citation_correctness_results.json")
    g = groundedness.get("summary", {})
    c = citations.get("summary", {})
    values = {
        "Groundedness": g.get("mean_groundedness", 0),
        "Citation precision": c.get("mean_citation_precision", 0),
        "Citation completeness": c.get("mean_citation_completeness", 0),
        "Refusal accuracy": g.get("refusal_accuracy", 0) or 0,
    }
    frame = pd.Series(values)
    ax = frame.plot(kind="bar", figsize=(10, 6))
    ax.set_title("RAG Answer Quality")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUTS / "rag_answer_quality.png", dpi=180)
    plt.close(ax.figure)


def latency_chart() -> None:
    payload = load_json("response_latency_results.json")
    records = payload.get("records", [])
    if not records:
        return
    frame = pd.DataFrame(records)
    columns = [column for column in ["query_embedding_ms", "retrieval_ms", "generation_ms", "total_ms"] if column in frame]
    ax = frame[columns].boxplot(figsize=(10, 6), showfliers=False)
    ax.set_title("Local RAG Latency Distribution")
    ax.set_ylabel("Milliseconds")
    ax.tick_params(axis="x", rotation=15)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUTS / "response_latency_distribution.png", dpi=180)
    plt.close(ax.figure)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    retrieval_chart()
    answer_quality_chart()
    latency_chart()
    print("Saved evaluation charts in outputs/.")


if __name__ == "__main__":
    main()
