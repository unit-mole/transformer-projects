from __future__ import annotations

import csv
from pathlib import Path

from src.experiment2_comparison import compare_experiment_runs


def _write_comparison(path: Path, answer: str, score: float, flag: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "id": "benchmark_1",
        "category": "Concept explanation",
        "difficulty": "intermediate",
        "topic": "classification",
        "prompt": "Explain classification.",
        "reference_answer": "Classification predicts categories.",
        "base_answer": "Base",
        "lora_answer": answer,
        "lora_hallucination_flag": flag,
        "lora_instruction_adherence": score,
        "lora_quality_rubric_score": score,
        "lora_rouge_l_f1": score,
        "lora_semantic_similarity": score,
        "lora_bertscore_f1": score,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def test_compare_experiment_runs_creates_human_review_file(tmp_path: Path) -> None:
    exp1 = tmp_path / "exp1" / "evaluation"
    exp2 = tmp_path / "exp2" / "evaluation"
    _write_comparison(exp1 / "comparison" / "per_example_base_vs_lora.csv", "First", 0.4, True)
    _write_comparison(exp2 / "comparison" / "per_example_base_vs_lora.csv", "Second", 0.8, False)
    output = tmp_path / "comparison"
    result = compare_experiment_runs(
        experiment1_evaluation_dir=exp1,
        experiment2_evaluation_dir=exp2,
        output_dir=output,
    )
    assert result["benchmark_examples"] == 1
    assert result["metric_comparison"]["bertscore_f1"]["mean_delta"] == 0.4
    review_file = output / "experiment1_vs_experiment2_per_example.csv"
    assert review_file.exists()
    with review_file.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["human_preferred_model"] == ""
    assert rows[0]["experiment2_answer"] == "Second"
