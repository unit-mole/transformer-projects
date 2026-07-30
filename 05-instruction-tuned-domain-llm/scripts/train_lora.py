#!/usr/bin/env python
"""Train the RTX-optimized FLAN-T5 LoRA adapter and save all artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LoraTrainingConfig, ModelConfig
from src.hardware_utils import detect_hardware
from src.model_training import train_lora_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "ml_ds_instruction_dataset_v2.jsonl"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "experiments" / "flan_t5_base_lora"))
    parser.add_argument("--base-model", default="auto", choices=["auto", "google/flan-t5-small", "google/flan-t5-base"])
    parser.add_argument("--epochs", type=float, default=6.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    hardware = detect_hardware()
    base_model = hardware.recommended_model_id if args.base_model == "auto" else args.base_model
    model_config = ModelConfig(base_model_id=base_model)
    training_config = replace(
        LoraTrainingConfig(),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    metadata = train_lora_adapter(
        args.dataset,
        args.output_dir,
        model_config=model_config,
        training_config=training_config,
        hardware_profile=hardware,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
