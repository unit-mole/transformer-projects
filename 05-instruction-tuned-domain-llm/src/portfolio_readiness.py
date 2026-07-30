"""Evidence-based portfolio readiness checklist for Project 05.

The score describes completion of the project evidence, not an intrinsic model
quality score. A high score still requires honest interpretation of metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .data_preprocessing import load_jsonl


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def assess_portfolio_readiness(project_root: str | Path) -> Dict[str, Any]:
    root = Path(project_root)
    checks: List[Dict[str, Any]] = []

    def add(name: str, weight: float, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "weight": weight, "passed": bool(passed), "evidence": evidence})

    # 1. Real Transformer + PEFT implementation.
    training_code = (root / "src" / "model_training.py").read_text(encoding="utf-8") if (root / "src" / "model_training.py").exists() else ""
    add(
        "Transformer and LoRA implementation",
        1.0,
        all(term in training_code for term in ("AutoModelForSeq2SeqLM", "get_peft_model", "Seq2SeqTrainer")),
        "src/model_training.py",
    )

    # 2. Dataset scale and split evidence.
    dataset_path = root / "data" / "ml_ds_instruction_dataset_v2.jsonl"
    dataset = load_jsonl(dataset_path) if dataset_path.exists() else []
    split_counts = {name: sum(str(row.get("split")) == name for row in dataset) for name in ("train", "validation", "test")}
    add(
        "Reviewed dataset scale",
        1.0,
        len(dataset) >= 450 and split_counts["validation"] >= 40 and split_counts["test"] >= 40,
        f"{len(dataset)} records; splits={split_counts}",
    )

    quality_report = _load_json(root / "outputs" / "dataset_generation" / "enhanced_dataset_quality_report.json")
    # An experiment-specific report may exist even before promotion.
    if not quality_report:
        candidates = sorted((root / "outputs" / "experiments").glob("*/dataset_generation/enhanced_dataset_quality_report.json")) if (root / "outputs" / "experiments").exists() else []
        quality_report = _load_json(candidates[-1]) if candidates else {}
    add(
        "Dataset validation, de-duplication, and leakage guard",
        1.0,
        bool(quality_report) and "benchmark_leakage_removed" in quality_report,
        "enhanced_dataset_quality_report.json",
    )

    adapter_dir = root / "models" / "lora_adapter"
    adapter_ready = (adapter_dir / "adapter_config.json").exists() and (adapter_dir / "adapter_model.safetensors").exists()
    add("Trained LoRA adapter", 1.5, adapter_ready, "models/lora_adapter")

    model_metadata = _load_json(root / "models" / "model_metadata.json")
    reproducible = bool(model_metadata.get("hardware")) and bool(model_metadata.get("training_config")) and bool(model_metadata.get("train_metrics"))
    add("Reproducible training metadata", 1.0, reproducible, "models/model_metadata.json")

    benchmark_path = root / "data" / "benchmark_prompts_v2.jsonl"
    benchmark = load_jsonl(benchmark_path) if benchmark_path.exists() else []
    add("Independent held-out benchmark", 1.0, len(benchmark) >= 80, f"{len(benchmark)} benchmark records")

    base_metrics = _load_json(root / "outputs" / "base_model_metrics.json")
    lora_metrics = _load_json(root / "outputs" / "lora_model_metrics.json")
    required_metric_keys = {
        "average_instruction_adherence",
        "average_quality_rubric_score",
        "average_rouge_l_f1",
        "average_semantic_similarity",
        "average_bertscore_f1",
    }
    metrics_ready = base_metrics.get("status") == "completed" and lora_metrics.get("status") == "completed" and required_metric_keys.issubset(lora_metrics)
    add("Multi-metric base and LoRA evaluation", 1.5, metrics_ready, "outputs/base_model_metrics.json and lora_model_metrics.json")

    comparison = _load_json(root / "outputs" / "base_vs_lora_comparison.json")
    comparison_ready = comparison.get("status") == "completed" and comparison.get("benchmark_examples", 0) >= 80
    add("Base-versus-LoRA comparison with confidence intervals", 1.0, comparison_ready, "outputs/base_vs_lora_comparison.json")

    release_manifest = _load_json(root / "outputs" / "release_manifest.json")
    human_review = release_manifest.get("human_review_completed") is True
    add("Human factual and preference review", 0.5, human_review, "outputs/release_manifest.json")

    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""
    no_placeholders = "YOUR_HUGGINGFACE" not in readme and "<your-" not in readme.lower()
    add("Published model/demo links and deployment evidence", 0.5, no_placeholders and adapter_ready, "README.md and Hugging Face deployment")

    score = round(sum(item["weight"] for item in checks if item["passed"]), 2)
    result = {
        "portfolio_readiness_score_out_of_10": score,
        "meaning": "Project evidence-completion score, not a universal model-quality score.",
        "target_reached": score >= 9.0,
        "checks": checks,
        "next_actions": [item["name"] for item in checks if not item["passed"]],
    }
    return result
