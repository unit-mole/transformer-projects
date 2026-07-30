"""Archive Experiment 1 without deleting its original output folder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_archive import archive_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        default=str(PROJECT_ROOT / "outputs" / "experiments" / "flan_t5_base_lora_20260730_120312"),
    )
    parser.add_argument(
        "--archive-root",
        default=str(PROJECT_ROOT / "outputs" / "experiment_archives"),
    )
    parser.add_argument("--label", default="experiment_1_initial_lora")
    parser.add_argument("--no-full-zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = archive_experiment(
        args.experiment_dir,
        args.archive_root,
        archive_label=args.label,
        create_full_zip=not args.no_full_zip,
        extra_files=[PROJECT_ROOT / "notebooks" / "05_full_training_evaluation_pipeline.ipynb"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
