#!/usr/bin/env python
"""Generate and validate the expanded ML/Data Science instruction dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DatasetGenerationConfig
from src.dataset_expansion import generate_enhanced_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dataset", default=str(PROJECT_ROOT / "data" / "ml_ds_instruction_dataset.jsonl"))
    parser.add_argument("--topic-plan", default=str(PROJECT_ROOT / "data" / "dataset_generation_plan.json"))
    parser.add_argument("--benchmark", default=str(PROJECT_ROOT / "data" / "benchmark_prompts_v2.jsonl"))
    parser.add_argument("--output-dataset", default=str(PROJECT_ROOT / "data" / "ml_ds_instruction_dataset_v2.jsonl"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "dataset_generation"))
    parser.add_argument("--target-examples", type=int, default=600)
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    config = DatasetGenerationConfig(
        teacher_model_id=args.teacher_model,
        target_examples=args.target_examples,
    )
    report = generate_enhanced_dataset(
        seed_dataset_path=args.seed_dataset,
        topic_plan_path=args.topic_plan,
        benchmark_path=args.benchmark,
        output_dataset_path=args.output_dataset,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
