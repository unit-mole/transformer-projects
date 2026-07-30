"""Build the Version 3 curated dataset for Experiment 2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment2_dataset import build_dataset_v3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-teacher-records", action="store_true")
    parser.add_argument("--seed", type=int, default=52)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset_v3(
        seed_dataset_path=PROJECT_ROOT / "data" / "ml_ds_instruction_dataset.jsonl",
        previous_v2_path=PROJECT_ROOT / "data" / "ml_ds_instruction_dataset_v2.jsonl",
        benchmark_path=PROJECT_ROOT / "data" / "benchmark_prompts_v2.jsonl",
        topic_cards_path=PROJECT_ROOT / "data" / "curated_topic_cards_v3.json",
        comparisons_path=PROJECT_ROOT / "data" / "curated_comparisons_v3.json",
        code_examples_path=PROJECT_ROOT / "data" / "curated_code_examples_v3.json",
        workflows_path=PROJECT_ROOT / "data" / "curated_workflows_v3.json",
        rules_path=PROJECT_ROOT / "data" / "experiment2_quality_rules.json",
        output_dataset_path=PROJECT_ROOT / "data" / "ml_ds_instruction_dataset_v3.jsonl",
        output_report_dir=PROJECT_ROOT / "outputs" / "experiment2_dataset_build",
        reuse_teacher_records=args.reuse_teacher_records,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
