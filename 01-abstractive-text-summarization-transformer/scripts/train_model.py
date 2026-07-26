from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_loader import load_public_dataset
from src.model_training import TrainingConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optionally fine-tune DistilBART.")
    parser.add_argument("--dataset", choices=["xsum", "cnn_dailymail"], default="xsum")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_frame = load_public_dataset(
        args.dataset, split="train", max_samples=args.train_samples
    )
    validation_split = "validation" if args.dataset == "xsum" else "validation"
    validation_frame = load_public_dataset(
        args.dataset, split=validation_split, max_samples=args.validation_samples
    )
    config = TrainingConfig(epochs=args.epochs)
    metrics = train_model(train_frame, validation_frame, config)
    output = PROJECT_ROOT / "outputs" / "training_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Training completed. Metrics: {output}")


if __name__ == "__main__":
    main()
