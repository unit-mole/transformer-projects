"""Compare Experiment 1 and Experiment 2 on the unchanged held-out benchmark."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], field: str) -> float:
    try:
        return float(row.get(field, "") or 0.0)
    except ValueError:
        return 0.0


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def compare_experiment_runs(
    *,
    experiment1_evaluation_dir: str | Path,
    experiment2_evaluation_dir: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    exp1_path = Path(experiment1_evaluation_dir) / "comparison" / "per_example_base_vs_lora.csv"
    exp2_path = Path(experiment2_evaluation_dir) / "comparison" / "per_example_base_vs_lora.csv"
    if not exp1_path.exists():
        raise FileNotFoundError(f"Experiment 1 comparison not found: {exp1_path}")
    if not exp2_path.exists():
        raise FileNotFoundError(f"Experiment 2 comparison not found: {exp2_path}")

    exp1 = {row["id"]: row for row in _read_csv(exp1_path)}
    exp2 = {row["id"]: row for row in _read_csv(exp2_path)}
    common_ids = sorted(exp1.keys() & exp2.keys())
    metrics = [
        "instruction_adherence",
        "quality_rubric_score",
        "rouge_l_f1",
        "semantic_similarity",
        "bertscore_f1",
    ]
    rows: list[dict[str, Any]] = []
    for record_id in common_ids:
        one, two = exp1[record_id], exp2[record_id]
        row: Dict[str, Any] = {
            "id": record_id,
            "category": two.get("category", one.get("category", "")),
            "difficulty": two.get("difficulty", one.get("difficulty", "")),
            "topic": two.get("topic", one.get("topic", "")),
            "prompt": two.get("prompt", one.get("prompt", "")),
            "reference_answer": two.get("reference_answer", one.get("reference_answer", "")),
            "base_answer": two.get("base_answer", one.get("base_answer", "")),
            "experiment1_answer": one.get("lora_answer", ""),
            "experiment2_answer": two.get("lora_answer", ""),
            "experiment1_hallucination_flag": one.get("lora_hallucination_flag", ""),
            "experiment2_hallucination_flag": two.get("lora_hallucination_flag", ""),
            "human_preferred_model": "",
            "human_notes": "",
        }
        for metric in metrics:
            exp1_value = _float(one, f"lora_{metric}")
            exp2_value = _float(two, f"lora_{metric}")
            row[f"experiment1_{metric}"] = exp1_value
            row[f"experiment2_{metric}"] = exp2_value
            row[f"delta_exp2_minus_exp1_{metric}"] = round(exp2_value - exp1_value, 6)
        rows.append(row)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "experiment1_vs_experiment2_per_example.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    metric_summary: Dict[str, Any] = {}
    for metric in metrics:
        exp1_values = [float(row[f"experiment1_{metric}"]) for row in rows]
        exp2_values = [float(row[f"experiment2_{metric}"]) for row in rows]
        deltas = [float(row[f"delta_exp2_minus_exp1_{metric}"]) for row in rows]
        metric_summary[metric] = {
            "experiment1_mean": _mean(exp1_values),
            "experiment2_mean": _mean(exp2_values),
            "mean_delta": _mean(deltas),
            "experiment2_win_rate": round(sum(value > 0 for value in deltas) / len(deltas), 6) if deltas else None,
            "tie_rate": round(sum(value == 0 for value in deltas) / len(deltas), 6) if deltas else None,
        }

    def as_bool(value: Any) -> bool:
        return str(value).strip().lower() in {"true", "1", "yes"}

    exp1_hallucination = [as_bool(row["experiment1_hallucination_flag"]) for row in rows]
    exp2_hallucination = [as_bool(row["experiment2_hallucination_flag"]) for row in rows]

    category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category_groups[str(row["category"])].append(row)
    category_summary: Dict[str, Any] = {}
    for category, group in sorted(category_groups.items()):
        category_summary[category] = {
            metric: {
                "experiment1": _mean([float(r[f"experiment1_{metric}"]) for r in group]),
                "experiment2": _mean([float(r[f"experiment2_{metric}"]) for r in group]),
                "delta": _mean([float(r[f"delta_exp2_minus_exp1_{metric}"]) for r in group]),
            }
            for metric in metrics
        }

    summary = {
        "status": "completed",
        "benchmark_examples": len(rows),
        "metric_comparison": metric_summary,
        "experiment1_hallucination_flag_rate": round(sum(exp1_hallucination) / len(rows), 6) if rows else None,
        "experiment2_hallucination_flag_rate": round(sum(exp2_hallucination) / len(rows), 6) if rows else None,
        "hallucination_flag_rate_delta_exp2_minus_exp1": (
            round((sum(exp2_hallucination) - sum(exp1_hallucination)) / len(rows), 6)
            if rows else None
        ),
        "by_category": category_summary,
        "human_review_required": True,
        "human_review_file": str(csv_path.resolve()),
        "interpretation": "Positive metric deltas favor Experiment 2. Promotion still requires human factual review.",
    }
    (output / "experiment1_vs_experiment2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
        labels = ["Adherence", "Quality", "ROUGE-L", "Semantic", "BERTScore"]
        exp1_means = [metric_summary[m]["experiment1_mean"] or 0 for m in metrics]
        exp2_means = [metric_summary[m]["experiment2_mean"] or 0 for m in metrics]
        x = list(range(len(metrics)))
        width = 0.36
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar([v - width / 2 for v in x], exp1_means, width, label="Experiment 1")
        ax.bar([v + width / 2 for v in x], exp2_means, width, label="Experiment 2")
        ax.set_xticks(x, labels, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Experiment 1 vs Experiment 2 — Same Held-Out Benchmark")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "experiment1_vs_experiment2_metrics.png", dpi=170)
        plt.close(fig)
    except ImportError:
        pass

    return summary


def assess_experiment2_release(
    *,
    experiment2_evaluation_dir: str | Path,
    experiment_comparison_dir: str | Path,
) -> Dict[str, Any]:
    """Apply transparent portfolio thresholds after human review files are completed."""
    evaluation = Path(experiment2_evaluation_dir)
    comparison_dir = Path(experiment_comparison_dir)
    exp2_review_path = evaluation / "lora_model" / "manual_review_results.csv"
    exp12_review_path = comparison_dir / "experiment1_vs_experiment2_per_example.csv"
    if not exp2_review_path.exists() or not exp12_review_path.exists():
        return {"ready": False, "reason": "Required review files are missing."}

    exp2_review = _read_csv(exp2_review_path)
    exp12_review = _read_csv(exp12_review_path)

    rating_fields = [
        "human_factuality_1_to_5",
        "human_relevance_1_to_5",
        "human_clarity_1_to_5",
        "human_instruction_following_1_to_5",
    ]
    rating_means = {
        field: _mean([_float(row, field) for row in exp2_review if str(row.get(field, "")).strip()])
        for field in rating_fields
    }
    preferences = {"experiment1": 0, "experiment2": 0, "tie": 0, "missing": 0}
    for row in exp12_review:
        value = str(row.get("human_preferred_model", "")).strip().lower()
        if value in preferences:
            preferences[value] += 1
        else:
            preferences["missing"] += 1
    reviewed = sum(preferences[key] for key in ("experiment1", "experiment2", "tie"))
    exp2_rate = preferences["experiment2"] / reviewed if reviewed else 0.0
    human_hallucinations = sum(
        str(row.get("human_hallucination_flag", "")).strip().lower() in {"true", "1", "yes"}
        for row in exp2_review
    )
    hallucination_rate = human_hallucinations / len(exp2_review) if exp2_review else 1.0

    checks = {
        "all_exp2_ratings_completed": all(
            all(str(row.get(field, "")).strip() for field in rating_fields)
            for row in exp2_review
        ),
        "all_exp1_vs_exp2_preferences_completed": preferences["missing"] == 0 and reviewed > 0,
        "experiment2_preference_rate_at_least_60_percent": exp2_rate >= 0.60,
        "mean_factuality_at_least_4": (rating_means["human_factuality_1_to_5"] or 0) >= 4.0,
        "mean_relevance_at_least_4": (rating_means["human_relevance_1_to_5"] or 0) >= 4.0,
        "mean_clarity_at_least_4": (rating_means["human_clarity_1_to_5"] or 0) >= 4.0,
        "mean_instruction_following_at_least_4": (rating_means["human_instruction_following_1_to_5"] or 0) >= 4.0,
        "human_hallucination_rate_below_10_percent": hallucination_rate < 0.10,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "experiment2_rating_means": rating_means,
        "experiment1_vs_experiment2_preferences": preferences,
        "experiment2_preference_rate": round(exp2_rate, 4),
        "experiment2_human_hallucination_rate": round(hallucination_rate, 4),
        "decision": "promote" if all(checks.values()) else "do_not_promote_yet",
    }
