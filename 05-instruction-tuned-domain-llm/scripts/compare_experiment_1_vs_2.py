"""Create an Experiment 1 versus Experiment 2 comparison and review file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment2_comparison import compare_experiment_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment1-dir",
        default=str(PROJECT_ROOT / "outputs" / "experiments" / "flan_t5_base_lora_20260730_120312"),
    )
    parser.add_argument("--experiment2-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp1 = Path(args.experiment1_dir)
    exp2 = Path(args.experiment2_dir)
    result = compare_experiment_runs(
        experiment1_evaluation_dir=exp1 / "evaluation",
        experiment2_evaluation_dir=exp2 / "evaluation",
        output_dir=exp2 / "evaluation" / "experiment1_vs_experiment2",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
