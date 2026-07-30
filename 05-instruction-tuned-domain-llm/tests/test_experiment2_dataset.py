from __future__ import annotations

import json
from pathlib import Path

from src.experiment2_dataset import build_dataset_v3


def test_build_dataset_v3_uses_curated_records_and_keeps_benchmark_held_out(tmp_path: Path, project_root: Path) -> None:
    output_dataset = tmp_path / "dataset_v3.jsonl"
    report_dir = tmp_path / "report"
    result = build_dataset_v3(
        seed_dataset_path=project_root / "data" / "ml_ds_instruction_dataset.jsonl",
        previous_v2_path=tmp_path / "missing_v2.jsonl",
        benchmark_path=project_root / "data" / "benchmark_prompts_v2.jsonl",
        topic_cards_path=project_root / "data" / "curated_topic_cards_v3.json",
        comparisons_path=project_root / "data" / "curated_comparisons_v3.json",
        code_examples_path=project_root / "data" / "curated_code_examples_v3.json",
        workflows_path=project_root / "data" / "curated_workflows_v3.json",
        rules_path=project_root / "data" / "experiment2_quality_rules.json",
        output_dataset_path=output_dataset,
        output_report_dir=report_dir,
        reuse_teacher_records=False,
        seed=52,
    )
    assert result["status"] == "completed"
    assert result["report"]["final_records"] >= 400
    assert result["report"]["previous_v2_records_reused"] == 0
    assert result["report"]["train_records"] > result["report"]["validation_records"] > 0
    assert result["report"]["test_records"] > 0
    assert output_dataset.exists()
    records = [json.loads(line) for line in output_dataset.read_text(encoding="utf-8").splitlines()]
    benchmark = [json.loads(line) for line in (project_root / "data" / "benchmark_prompts_v2.jsonl").read_text(encoding="utf-8").splitlines()]
    benchmark_prompts = {str(row.get("instruction") or row.get("prompt")).strip().lower() for row in benchmark}
    assert not any(record["instruction"].strip().lower() in benchmark_prompts for record in records)
    assert {record["split"] for record in records} == {"train", "validation", "test"}


def test_curated_outputs_avoid_known_circular_phrases(project_root: Path) -> None:
    files = [
        project_root / "data" / "curated_topic_cards_v3.json",
        project_root / "data" / "curated_comparisons_v3.json",
        project_root / "data" / "curated_code_examples_v3.json",
        project_root / "data" / "curated_workflows_v3.json",
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in files)
    assert "classification is a classification" not in text
    assert "outliers are a set of outliers" not in text
    assert "learns a large number of data points at a time" not in text
