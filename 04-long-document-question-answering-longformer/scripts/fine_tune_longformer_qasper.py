from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.qasper_dataset import load_prepared_split
from src.qasper_training import PROFILES, fine_tune_longformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Longformer on the extractive QASPER subset.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="portfolio")
    parser.add_argument(
        "--base-model",
        default="valhalla/longformer-base-4096-finetuned-squadv1",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = PROJECT_ROOT / "data" / "processed" / "qasper"
    train_path = data_dir / "qasper_train_extractive.parquet"
    validation_path = data_dir / "qasper_validation_extractive.parquet"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Run scripts/prepare_qasper_dataset.py first.")
    summary = fine_tune_longformer(
        load_prepared_split(train_path),
        load_prepared_split(validation_path),
        PROJECT_ROOT,
        profile_name=args.profile,
        base_model_id=args.base_model,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
