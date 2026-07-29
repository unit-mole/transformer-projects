from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.training import FineTuneConfig, fine_tune_bi_encoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune MiniLM on SciFact with BM25 hard negatives."
    )
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-name", default="docrank360-minilm-scifact")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--hard-negative-pool", type=int, default=50)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    args = parser.parse_args()

    metadata = fine_tune_bi_encoder(
        FineTuneConfig(
            project_root=PROJECT_ROOT,
            base_model=args.base_model,
            output_name=args.output_name,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            hard_negative_pool=args.hard_negative_pool,
            max_train_pairs=args.max_train_pairs,
        )
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
