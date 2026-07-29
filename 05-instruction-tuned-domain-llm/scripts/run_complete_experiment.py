#!/usr/bin/env python
"""Run dataset build, GPU LoRA training, base-vs-adapter evaluation, and result synchronization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.build_extended_dataset import build_dataset
from src.advanced_evaluation import EvaluationConfig, run_base_vs_lora_evaluation
from src.advanced_training import plot_training_history, train_portfolio_lora
from src.experiment_utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "portfolio_experiment.yaml")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    if not args.skip_dataset:
        build_dataset(
            PROJECT_DIR / "data" / "ml_ds_instruction_dataset.jsonl",
            PROJECT_DIR / data_cfg["dataset_path"],
            PROJECT_DIR / data_cfg["evaluation_path"],
            PROJECT_DIR / "outputs" / "extended_dataset_statistics.json",
            PROJECT_DIR / "outputs" / "extended_dataset_validation_report.json",
        )

    if not args.skip_training:
        training = train_portfolio_lora(args.config, resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        training = load_json(PROJECT_DIR / train_cfg["output_dir"] / "latest_training_metadata.json")
        if not training:
            raise FileNotFoundError("No latest training metadata found. Run training or remove --skip-training.")

    run_dir = Path(training["adapter_path"]).parent
    history = load_json(run_dir / "training_log_history.json", default=[])
    plot_training_history(history, PROJECT_DIR / "outputs" / "training_curve.png")

    results = {"training": training}
    if not args.skip_evaluation:
        evaluation_config = EvaluationConfig(
            max_source_length=int(eval_cfg["max_source_length"]),
            max_target_length=int(eval_cfg["max_target_length"]),
            max_new_tokens=int(eval_cfg["max_new_tokens"]),
            num_beams=int(eval_cfg["num_beams"]),
            repetition_penalty=float(eval_cfg["repetition_penalty"]),
            batch_size_loss=int(eval_cfg["batch_size_loss"]),
            seed=int(train_cfg["seed"]),
            bertscore_model_type=str(eval_cfg["bertscore_model_type"]),
            embedding_model_id=str(eval_cfg["embedding_model_id"]),
            bootstrap_samples=int(eval_cfg["bootstrap_samples"]),
            confidence_level=float(eval_cfg["confidence_level"]),
            low_reference_support_threshold=float(eval_cfg["low_reference_support_threshold"]),
        )
        results["evaluation"] = run_base_vs_lora_evaluation(
            base_model_id=config["model"]["base_model_id"],
            adapter_path=training["adapter_path"],
            evaluation_path=PROJECT_DIR / data_cfg["evaluation_path"],
            output_dir=PROJECT_DIR / eval_cfg["output_dir"],
            config=evaluation_config,
        )
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
