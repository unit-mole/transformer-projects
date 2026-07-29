#!/usr/bin/env python
"""Synchronize real evaluation results into the static demo and Markdown snippets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    return "Not run" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=PROJECT_DIR / "outputs" / "portfolio_experiment" / "model_metrics.json")
    parser.add_argument("--dataset-stats", type=Path, default=PROJECT_DIR / "outputs" / "extended_dataset_statistics.json")
    parser.add_argument("--web-output", type=Path, default=PROJECT_DIR / "web" / "public" / "evaluation-summary.json")
    parser.add_argument("--snippet", type=Path, default=PROJECT_DIR / "outputs" / "README_RESULTS_SNIPPET.md")
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    stats = load_json(args.dataset_stats)
    base = metrics["base_model"]
    lora = metrics["lora_model"]
    comparison = metrics["comparison"]
    web_payload = {
        "status": "completed",
        "dataset_examples": stats["rows"],
        "held_out_examples": lora["evaluated_examples"],
        "metrics": {
            "Dataset examples": str(stats["rows"]),
            "Held-out prompts": str(lora["evaluated_examples"]),
            "Base BERTScore F1": fmt(base["mean_bertscore_f1"]),
            "LoRA BERTScore F1": fmt(lora["mean_bertscore_f1"]),
            "Base semantic relevance": fmt(base["mean_semantic_relevance"]),
            "LoRA semantic relevance": fmt(lora["mean_semantic_relevance"]),
            "Base adherence": fmt(base["mean_adherence_score"]),
            "LoRA adherence": fmt(lora["mean_adherence_score"]),
            "LoRA mean latency (s)": fmt(lora["mean_latency_seconds"]),
        },
        "base_model": {
            "bertscore_f1": base["mean_bertscore_f1"],
            "rouge_l": base["mean_rougeL"],
            "semantic_relevance": base["mean_semantic_relevance"],
            "instruction_adherence": base["mean_adherence_score"],
            "hallucination_risk_flag_rate": base["hallucination_risk_flag_rate"],
            "latency_seconds": base["mean_latency_seconds"],
            "heldout_loss": base["heldout_loss"],
            "perplexity": base["perplexity"],
        },
        "lora_model": {
            "bertscore_f1": lora["mean_bertscore_f1"],
            "rouge_l": lora["mean_rougeL"],
            "semantic_relevance": lora["mean_semantic_relevance"],
            "instruction_adherence": lora["mean_adherence_score"],
            "hallucination_risk_flag_rate": lora["hallucination_risk_flag_rate"],
            "latency_seconds": lora["mean_latency_seconds"],
            "heldout_loss": lora["heldout_loss"],
            "perplexity": lora["perplexity"],
        },
        "paired_improvement": {
            "bertscore_f1": comparison["metrics"]["bertscore_f1"],
            "semantic_relevance": comparison["metrics"]["semantic_relevance"],
            "instruction_adherence": comparison["metrics"]["adherence_score"],
        },
        "disclaimer": "Automated metrics are quality proxies and require human factuality review.",
    }
    args.web_output.parent.mkdir(parents=True, exist_ok=True)
    args.web_output.write_text(json.dumps(web_payload, indent=2), encoding="utf-8")

    delta = comparison["metrics"]["bertscore_f1"]
    lines = [
        "## Executed experiment results",
        "",
        f"The final run used **{stats['rows']}** public-safe instruction examples with topic-grouped splits and evaluated **{lora['evaluated_examples']}** held-out prompts.",
        "",
        "| Metric | Base FLAN-T5 | LoRA adapter |",
        "|---|---:|---:|",
        f"| Held-out loss | {fmt(base['heldout_loss'])} | {fmt(lora['heldout_loss'])} |",
        f"| Perplexity | {fmt(base['perplexity'])} | {fmt(lora['perplexity'])} |",
        f"| BERTScore F1 | {fmt(base['mean_bertscore_f1'])} | {fmt(lora['mean_bertscore_f1'])} |",
        f"| ROUGE-L | {fmt(base['mean_rougeL'])} | {fmt(lora['mean_rougeL'])} |",
        f"| Semantic relevance | {fmt(base['mean_semantic_relevance'])} | {fmt(lora['mean_semantic_relevance'])} |",
        f"| Instruction adherence | {fmt(base['mean_adherence_score'])} | {fmt(lora['mean_adherence_score'])} |",
        f"| Automated hallucination-risk flag rate | {fmt(base['hallucination_risk_flag_rate'])} | {fmt(lora['hallucination_risk_flag_rate'])} |",
        f"| Mean warm-cache latency, seconds | {fmt(base['mean_latency_seconds'])} | {fmt(lora['mean_latency_seconds'])} |",
        "",
        f"Paired BERTScore F1 improvement: **{delta['mean_delta']:.4f}**, with a 95% bootstrap interval of **[{delta['ci_low']:.4f}, {delta['ci_high']:.4f}]**.",
        "",
        "> BERTScore, ROUGE, embedding similarity, and heuristic risk flags do not prove factual correctness. See the per-example CSV files and complete the manual review template.",
    ]
    args.snippet.parent.mkdir(parents=True, exist_ok=True)
    args.snippet.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"web_output": str(args.web_output), "snippet": str(args.snippet)}, indent=2))


if __name__ == "__main__":
    main()
