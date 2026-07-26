from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_training import fine_tune_marian  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optionally fine-tune MarianMT.")
    parser.add_argument("--direction", required=True, choices=["en_hi", "hi_en"])
    parser.add_argument("--dataset", default="cfilt/iitb-english-hindi")
    parser.add_argument("--train-split", default="train[:10000]")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = fine_tune_marian(
        direction=args.direction,
        dataset_name=args.dataset,
        train_split=args.train_split,
        validation_split=args.validation_split,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(result, indent=2))
