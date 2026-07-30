#!/usr/bin/env python
"""One-command dataset generation, LoRA training, and base-vs-LoRA evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_evaluation import run_base_vs_lora_evaluation
from src.config import DatasetGenerationConfig, EvaluationConfig, LoraTrainingConfig, ModelConfig
from src.dataset_expansion import generate_enhanced_dataset
from src.hardware_utils import detect_hardware
from src.model_training import train_lora_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=datetime.now().strftime("flan_t5_lora_%Y%m%d_%H%M%S"))
    parser.add_argument("--skip-dataset-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--target-examples", type=int, default=600)
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--base-model", default="auto", choices=["auto", "google/flan-t5-small", "google/flan-t5-base"])
    parser.add_argument("--epochs", type=float, default=6.0)
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "outputs" / "experiments" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = PROJECT_ROOT / "data" / "ml_ds_instruction_dataset_v2.jsonl"
    benchmark_path = PROJECT_ROOT / "data" / "benchmark_prompts_v2.jsonl"
    manifest = {"run_name": args.run_name, "run_dir": str(run_dir), "stages": {}}

    if not args.skip_dataset_generation:
        dataset_report = generate_enhanced_dataset(
            seed_dataset_path=PROJECT_ROOT / "data" / "ml_ds_instruction_dataset.jsonl",
            topic_plan_path=PROJECT_ROOT / "data" / "dataset_generation_plan.json",
            benchmark_path=benchmark_path,
            output_dataset_path=dataset_path,
            output_dir=run_dir / "dataset_generation",
            config=DatasetGenerationConfig(
                teacher_model_id=args.teacher_model,
                target_examples=args.target_examples,
            ),
        )
        manifest["stages"]["dataset_generation"] = dataset_report

    hardware = detect_hardware()
    base_model = hardware.recommended_model_id if args.base_model == "auto" else args.base_model
    adapter_path = run_dir / "training" / "lora_adapter"

    if not args.skip_training:
        training_report = train_lora_adapter(
            dataset_path,
            run_dir / "training",
            model_config=ModelConfig(base_model_id=base_model),
            training_config=replace(LoraTrainingConfig(), num_train_epochs=args.epochs),
            hardware_profile=hardware,
        )
        manifest["stages"]["training"] = training_report

    if not args.skip_evaluation:
        evaluation_report = run_base_vs_lora_evaluation(
            benchmark_path=benchmark_path,
            base_model_id=base_model,
            adapter_path=adapter_path,
            output_dir=run_dir / "evaluation",
            evaluation_config=EvaluationConfig(),
        )
        manifest["stages"]["evaluation"] = evaluation_report

    (run_dir / "full_experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
