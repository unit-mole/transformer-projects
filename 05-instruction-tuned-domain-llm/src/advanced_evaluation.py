"""Portfolio-grade base-versus-LoRA evaluation and reporting.

The benchmark is held out from dataset generation. The same deterministic
prompts and generation settings are used for the base and adapted models.
Automated metrics are complemented by a manual-review CSV; no heuristic is
presented as proof of factual correctness.
"""
from __future__ import annotations

import csv
import gc
import json
import math
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .bertscore_evaluation import calculate_bertscore
from .config import EvaluationConfig, ModelConfig
from .data_preprocessing import load_jsonl
from .hallucination_analysis import analyze_hallucination_risk
from .inference_pipeline import InstructionAssistant
from .instruction_adherence import evaluate_instruction_adherence
from .relevance_scoring import score_relevance


def _safe_mean(values: Sequence[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def _safe_stdev(values: Sequence[float]) -> float | None:
    return round(statistics.stdev(values), 6) if len(values) > 1 else None


def _rouge_l(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, Any]:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError("Install rouge-score for ROUGE-L evaluation.") from exc
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows = [scorer.score(ref, pred)["rougeL"] for pred, ref in zip(predictions, references)]
    f1 = [round(float(row.fmeasure), 6) for row in rows]
    return {
        "precision": [round(float(row.precision), 6) for row in rows],
        "recall": [round(float(row.recall), 6) for row in rows],
        "f1": f1,
        "average_f1": _safe_mean(f1),
        "status": "completed",
    }


def _semantic_similarity(predictions: Sequence[str], references: Sequence[str], model_id: str) -> Dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers for semantic similarity evaluation.") from exc
    model = SentenceTransformer(model_id)
    prediction_embeddings = model.encode(list(predictions), normalize_embeddings=True, show_progress_bar=True)
    reference_embeddings = model.encode(list(references), normalize_embeddings=True, show_progress_bar=True)
    similarities = [
        round(float((pred * ref).sum()), 6)
        for pred, ref in zip(prediction_embeddings, reference_embeddings)
    ]
    del model
    gc.collect()
    return {
        "model_id": model_id,
        "scores": similarities,
        "average": _safe_mean(similarities),
        "status": "completed",
        "limitation": "embedding_similarity_is_not_factual_correctness",
    }


def response_quality_rubric(record: Dict[str, Any], response: str) -> Dict[str, Any]:
    """Transparent 0-1 rubric for completeness and requested format."""
    instruction = str(record.get("instruction", ""))
    category = str(record.get("category", ""))
    topic = str(record.get("topic", ""))
    lower_instruction = instruction.lower()
    lower_response = response.lower()
    words = response.split()

    length_score = 1.0 if 35 <= len(words) <= 260 else 0.7 if 20 <= len(words) <= 340 else 0.3
    topic_tokens = {t for t in re.findall(r"[a-z0-9+-]+", topic.lower()) if len(t) > 2}
    topic_score = 1.0 if not topic_tokens or topic_tokens & set(re.findall(r"[a-z0-9+-]+", lower_response)) else 0.4

    requested_caveat = any(term in lower_instruction for term in ("limitation", "caveat", "risk", "pitfall"))
    caveat_markers = ("however", "limitation", "caveat", "risk", "but", "should", "depends")
    caveat_score = 1.0 if not requested_caveat or any(marker in lower_response for marker in caveat_markers) else 0.3

    format_score = 1.0
    if category == "Algorithm comparison" or "compare" in lower_instruction:
        format_score = 1.0 if any(marker in lower_response for marker in ("whereas", "while", "both", "difference", "compared")) else 0.45
    elif category == "Small code example" or "python" in lower_instruction or "code" in lower_instruction:
        format_score = 1.0 if any(marker in response for marker in ("```", "import ", "def ", "=", "#")) else 0.35
    elif category in {"Data Science workflow", "ML project guidance"} or "step" in lower_instruction:
        format_score = 1.0 if re.search(r"(^|\n)\s*(?:\d+[.)]|[-*])\s+", response) else 0.55

    refusal = bool(re.search(r"\b(i cannot|i can't|unable to help|cannot assist)\b", lower_response))
    unsupported_certainty = bool(re.search(r"\b(always|never|guarantee[sd]?|100 percent|perfect)\b", lower_response))
    score = 0.3 * length_score + 0.25 * topic_score + 0.25 * format_score + 0.2 * caveat_score
    score -= 0.25 * float(refusal)
    score -= 0.1 * float(unsupported_certainty)
    return {
        "quality_rubric_score": round(max(0.0, min(1.0, score)), 4),
        "length_score": length_score,
        "topic_score": topic_score,
        "format_score": format_score,
        "caveat_score": caveat_score,
        "safe_prompt_refusal": refusal,
        "unsupported_certainty": unsupported_certainty,
        "method": "transparent_rule_based_rubric_requires_human_review",
    }


def _generation_loop(records: Sequence[Dict[str, Any]], model_config: ModelConfig, model_name: str) -> List[Dict[str, Any]]:
    assistant = InstructionAssistant(model_config)
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        started = time.perf_counter()
        result = assistant.generate(
            str(record.get("instruction", "")),
            str(record.get("category", "Concept explanation")),
            str(record.get("input", "")),
            max_new_tokens=220,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.12,
        )
        elapsed = time.perf_counter() - started
        response = str(result.get("response", ""))
        reference = str(record.get("reference_answer", record.get("output", "")))
        adherence = evaluate_instruction_adherence(str(record.get("instruction", "")), response)
        relevance = score_relevance(
            f"{record.get('instruction', '')} {record.get('input', '')}", response
        )
        hallucination = analyze_hallucination_risk(str(record.get("instruction", "")), response, reference)
        rubric = response_quality_rubric(record, response)
        rows.append({
            "id": record.get("id", f"example_{index:04d}"),
            "model_name": model_name,
            "prompt": record.get("instruction", ""),
            "input": record.get("input", ""),
            "category": record.get("category", ""),
            "difficulty": record.get("difficulty", ""),
            "topic": record.get("topic", ""),
            "reference_answer": reference,
            "generated_answer": response,
            "instruction_adherence": adherence["adherence_score"],
            "response_relevance_tfidf": relevance["combined_relevance"],
            "quality_rubric_score": rubric["quality_rubric_score"],
            "hallucination_flag": hallucination["hallucination_flag"],
            "hallucination_severity": hallucination["severity"],
            "hallucination_issue_types": hallucination["issue_types"],
            "latency_seconds": round(float(result.get("latency_seconds", elapsed)), 4),
            "response_words": len(response.split()),
            "model_mode": result.get("model_mode", "unknown"),
            "human_factuality_1_to_5": "",
            "human_relevance_1_to_5": "",
            "human_clarity_1_to_5": "",
            "human_instruction_following_1_to_5": "",
            "human_hallucination_flag": "",
            "reviewer_notes": "",
        })
    # Release the model before loading embedding metrics or the comparison model.
    del assistant
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return rows


def _add_reference_metrics(rows: List[Dict[str, Any]], config: EvaluationConfig) -> Dict[str, Any]:
    predictions = [str(row["generated_answer"]) for row in rows]
    references = [str(row["reference_answer"]) for row in rows]
    result: Dict[str, Any] = {}
    if config.include_rouge:
        rouge = _rouge_l(predictions, references)
        for row, value in zip(rows, rouge["f1"]):
            row["rouge_l_f1"] = value
        result["rouge_l"] = {key: value for key, value in rouge.items() if key not in {"precision", "recall", "f1"}}
    if config.include_semantic_similarity:
        semantic = _semantic_similarity(predictions, references, config.semantic_model_id)
        for row, value in zip(rows, semantic["scores"]):
            row["semantic_similarity"] = value
        result["semantic_similarity"] = {key: value for key, value in semantic.items() if key != "scores"}
    if config.include_bertscore:
        bertscore = calculate_bertscore(predictions, references, model_type=config.bertscore_model_type)
        for row, p, r, f in zip(rows, bertscore["precision"], bertscore["recall"], bertscore["f1"]):
            row["bertscore_precision"] = p
            row["bertscore_recall"] = r
            row["bertscore_f1"] = f
        result["bertscore"] = {
            key: value for key, value in bertscore.items() if key not in {"precision", "recall", "f1"}
        }
    return result


def _aggregate(rows: Sequence[Dict[str, Any]], reference_metrics: Dict[str, Any]) -> Dict[str, Any]:
    numeric_fields = [
        "instruction_adherence",
        "response_relevance_tfidf",
        "quality_rubric_score",
        "rouge_l_f1",
        "semantic_similarity",
        "bertscore_f1",
        "latency_seconds",
        "response_words",
    ]
    summary: Dict[str, Any] = {
        "status": "completed" if rows else "no_records",
        "evaluated_examples": len(rows),
        "hallucination_flag_rate": round(sum(bool(row["hallucination_flag"]) for row in rows) / len(rows), 6) if rows else None,
        "reference_metrics": reference_metrics,
        "warning": "Automated metrics and heuristic hallucination flags require human factual review.",
    }
    for field in numeric_fields:
        values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
        summary[f"average_{field}"] = _safe_mean(values)
        summary[f"stdev_{field}"] = _safe_stdev(values)

    category_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    difficulty_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category_rows[str(row["category"])].append(row)
        difficulty_rows[str(row["difficulty"])].append(row)

    def slice_summary(group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "count": len(group),
            "instruction_adherence": _safe_mean([float(r["instruction_adherence"]) for r in group]),
            "quality_rubric_score": _safe_mean([float(r["quality_rubric_score"]) for r in group]),
            "bertscore_f1": _safe_mean([float(r.get("bertscore_f1", 0.0)) for r in group]),
            "semantic_similarity": _safe_mean([float(r.get("semantic_similarity", 0.0)) for r in group]),
            "hallucination_flag_rate": round(sum(bool(r["hallucination_flag"]) for r in group) / len(group), 6),
        }

    summary["by_category"] = {name: slice_summary(group) for name, group in sorted(category_rows.items())}
    summary["by_difficulty"] = {name: slice_summary(group) for name, group in sorted(difficulty_rows.items())}
    return summary


def _write_rows(rows: Sequence[Dict[str, Any]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "per_example_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if rows:
        fieldnames = list(rows[0].keys())
        with (directory / "manual_review_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def evaluate_model(
    records: Sequence[Dict[str, Any]],
    *,
    model_config: ModelConfig,
    model_name: str,
    output_dir: str | Path,
    evaluation_config: EvaluationConfig | None = None,
) -> Dict[str, Any]:
    config = evaluation_config or EvaluationConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = _generation_loop(records, model_config, model_name)
    reference_metrics = _add_reference_metrics(rows, config)
    summary = _aggregate(rows, reference_metrics)
    _write_rows(rows, output)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"rows": rows, "summary": summary}


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float | int | None]:
    if not values:
        return {"mean": None, "lower": None, "upper": None, "n": 0}
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(statistics.mean(sample))
    estimates.sort()
    alpha = (1 - confidence_level) / 2
    lower_index = max(0, min(len(estimates) - 1, int(alpha * len(estimates))))
    upper_index = max(0, min(len(estimates) - 1, int((1 - alpha) * len(estimates)) - 1))
    return {
        "mean": round(statistics.mean(values), 6),
        "lower": round(estimates[lower_index], 6),
        "upper": round(estimates[upper_index], 6),
        "n": n,
    }


def _comparison_rows(base_rows: Sequence[Dict[str, Any]], lora_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_by_id = {str(row["id"]): row for row in base_rows}
    lora_by_id = {str(row["id"]): row for row in lora_rows}
    metrics = ["instruction_adherence", "quality_rubric_score", "rouge_l_f1", "semantic_similarity", "bertscore_f1"]
    rows: List[Dict[str, Any]] = []
    for record_id in sorted(base_by_id.keys() & lora_by_id.keys()):
        base, lora = base_by_id[record_id], lora_by_id[record_id]
        row: Dict[str, Any] = {
            "id": record_id,
            "category": base["category"],
            "difficulty": base["difficulty"],
            "topic": base["topic"],
            "prompt": base["prompt"],
            "reference_answer": base["reference_answer"],
            "base_answer": base["generated_answer"],
            "lora_answer": lora["generated_answer"],
            "base_hallucination_flag": base["hallucination_flag"],
            "lora_hallucination_flag": lora["hallucination_flag"],
            "base_latency_seconds": base["latency_seconds"],
            "lora_latency_seconds": lora["latency_seconds"],
            "human_preferred_model": "",
            "human_notes": "",
        }
        for metric in metrics:
            base_value = float(base.get(metric, 0.0) or 0.0)
            lora_value = float(lora.get(metric, 0.0) or 0.0)
            row[f"base_{metric}"] = base_value
            row[f"lora_{metric}"] = lora_value
            row[f"delta_{metric}"] = round(lora_value - base_value, 6)
        rows.append(row)
    return rows


def _plot_comparison(summary: Dict[str, Any], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metrics = ["instruction_adherence", "quality_rubric_score", "rouge_l_f1", "semantic_similarity", "bertscore_f1"]
    labels = ["Adherence", "Quality rubric", "ROUGE-L", "Semantic", "BERTScore"]
    base = [summary["metric_comparison"][m]["base_mean"] or 0 for m in metrics]
    lora = [summary["metric_comparison"][m]["lora_mean"] or 0 for m in metrics]
    x = list(range(len(metrics)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([v - width / 2 for v in x], base, width, label="Base FLAN-T5")
    ax.bar([v + width / 2 for v in x], lora, width, label="LoRA fine-tuned")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Base vs LoRA Evaluation on Held-Out Benchmark")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "base_vs_lora_metric_comparison.png", dpi=170)
    plt.close(fig)


def compare_models(
    base_result: Dict[str, Any],
    lora_result: Dict[str, Any],
    *,
    output_dir: str | Path,
    evaluation_config: EvaluationConfig | None = None,
) -> Dict[str, Any]:
    config = evaluation_config or EvaluationConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = _comparison_rows(base_result["rows"], lora_result["rows"])
    metrics = ["instruction_adherence", "quality_rubric_score", "rouge_l_f1", "semantic_similarity", "bertscore_f1"]
    metric_comparison: Dict[str, Any] = {}
    for metric in metrics:
        base_values = [float(row[f"base_{metric}"]) for row in rows]
        lora_values = [float(row[f"lora_{metric}"]) for row in rows]
        deltas = [float(row[f"delta_{metric}"]) for row in rows]
        metric_comparison[metric] = {
            "base_mean": _safe_mean(base_values),
            "lora_mean": _safe_mean(lora_values),
            "mean_delta": _safe_mean(deltas),
            "delta_95_percent_ci": bootstrap_mean_ci(
                deltas,
                samples=config.bootstrap_samples,
                confidence_level=config.confidence_level,
                seed=config.seed,
            ),
            "lora_win_rate": round(sum(delta > 0 for delta in deltas) / len(deltas), 6) if deltas else None,
            "tie_rate": round(sum(delta == 0 for delta in deltas) / len(deltas), 6) if deltas else None,
        }

    base_hallucination = [bool(row["base_hallucination_flag"]) for row in rows]
    lora_hallucination = [bool(row["lora_hallucination_flag"]) for row in rows]
    summary = {
        "status": "completed",
        "benchmark_examples": len(rows),
        "metric_comparison": metric_comparison,
        "base_hallucination_flag_rate": round(sum(base_hallucination) / len(rows), 6) if rows else None,
        "lora_hallucination_flag_rate": round(sum(lora_hallucination) / len(rows), 6) if rows else None,
        "hallucination_flag_rate_delta": round((sum(lora_hallucination) - sum(base_hallucination)) / len(rows), 6) if rows else None,
        "interpretation": "A positive metric delta favors LoRA. Confidence intervals quantify sampling uncertainty on this benchmark, not universal model quality.",
        "human_review_required": True,
    }

    if rows:
        with (output / "per_example_base_vs_lora.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (output / "base_vs_lora_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_comparison(summary, output)

    # Save a compact, recruiter-readable sample table.
    ranked = sorted(rows, key=lambda row: row["delta_bertscore_f1"] + row["delta_quality_rubric_score"], reverse=True)
    selected = ranked[:5] + ranked[-3:] if len(ranked) >= 8 else ranked
    lines = [
        "# Before vs After Fine-Tuning Examples",
        "",
        "> These examples are generated from the held-out benchmark. Human review is required before making factual claims.",
        "",
    ]
    for row in selected:
        lines.extend([
            f"## {row['id']} — {row['topic']}",
            "",
            f"**Prompt:** {row['prompt']}",
            "",
            f"**Base model:** {row['base_answer']}",
            "",
            f"**LoRA model:** {row['lora_answer']}",
            "",
            f"**BERTScore F1 delta:** {row['delta_bertscore_f1']:+.4f}",
            "",
        ])
    (output / "before_after_finetuning_examples.md").write_text("\n".join(lines), encoding="utf-8")
    return {"rows": rows, "summary": summary}


def run_base_vs_lora_evaluation(
    *,
    benchmark_path: str | Path,
    base_model_id: str,
    adapter_path: str | Path,
    output_dir: str | Path,
    evaluation_config: EvaluationConfig | None = None,
) -> Dict[str, Any]:
    """Run the complete held-out benchmark twice and save all comparison artifacts."""
    records = load_jsonl(benchmark_path)
    output = Path(output_dir)
    config = evaluation_config or EvaluationConfig()

    base_config = ModelConfig(base_model_id=base_model_id, adapter_id="", local_adapter_path=str(output / "no_adapter"))
    lora_config = ModelConfig(base_model_id=base_model_id, adapter_id="", local_adapter_path=str(adapter_path))

    base_result = evaluate_model(
        records,
        model_config=base_config,
        model_name="base_flan_t5",
        output_dir=output / "base_model",
        evaluation_config=config,
    )
    lora_result = evaluate_model(
        records,
        model_config=lora_config,
        model_name="lora_fine_tuned",
        output_dir=output / "lora_model",
        evaluation_config=config,
    )
    comparison = compare_models(base_result, lora_result, output_dir=output / "comparison", evaluation_config=config)
    manifest = {
        "status": "completed",
        "benchmark_path": str(benchmark_path),
        "base_model_id": base_model_id,
        "adapter_path": str(adapter_path),
        "base_metrics": base_result["summary"],
        "lora_metrics": lora_result["summary"],
        "comparison": comparison["summary"],
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
