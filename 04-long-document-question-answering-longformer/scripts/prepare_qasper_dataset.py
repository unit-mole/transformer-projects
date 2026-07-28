from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.qasper_dataset import prepare_qasper_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare the extractive QASPER subset.")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_qasper_dataset(
        PROJECT_ROOT,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        seed=args.seed,
        force_download=args.force_download,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
