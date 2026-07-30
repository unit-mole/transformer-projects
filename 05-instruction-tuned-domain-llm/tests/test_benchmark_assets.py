from __future__ import annotations

import json
from pathlib import Path


def test_benchmark_and_generation_plan_are_complete(project_root: Path) -> None:
    benchmark = [json.loads(line) for line in (project_root / "data" / "benchmark_prompts_v2.jsonl").read_text().splitlines() if line.strip()]
    plan = json.loads((project_root / "data" / "dataset_generation_plan.json").read_text())
    assert len(benchmark) == 80
    assert len({row["instruction"] for row in benchmark}) == 80
    assert all(row["reference_answer"] for row in benchmark)
    assert len(plan) == 64
