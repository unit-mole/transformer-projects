#!/usr/bin/env python
"""Evaluate base FLAN-T5 and the trained LoRA adapter on the same benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_evaluation import run_base_vs_lora_evaluation
from src.config import EvaluationConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=str(PROJECT_ROOT / "data" / "benchmark_prompts_v2.jsonl"))
    parser.add_argument("--base-model", default="google/flan-t5-base")
    parser.add_argument("--adapter", default=str(PROJECT_ROOT / "outputs" / "experiments" / "flan_t5_base_lora" / "lora_adapter"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "experiments" / "flan_t5_base_lora" / "evaluation"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--skip-bertscore", action="store_true")
    args = parser.parse_args()

    config = EvaluationConfig(
        benchmark_path=args.benchmark,
        bootstrap_samples=args.bootstrap_samples,
        include_bertscore=not args.skip_bertscore,
    )
    manifest = run_base_vs_lora_evaluation(
        benchmark_path=args.benchmark,
        base_model_id=args.base_model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        evaluation_config=config,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
