from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import preprocess_parallel_dataframe  # noqa: E402
from src.model_evaluation import evaluate_bidirectional, save_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate English↔Hindi MarianMT translation."
    )
    parser.add_argument("--input", required=True, help="Parallel CSV file.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw = pd.read_csv(args.input)
    dataframe = preprocess_parallel_dataframe(raw)
    if dataframe.empty:
        raise SystemExit("No valid English–Hindi pairs were found.")

    examples, metrics = evaluate_bidirectional(dataframe)
    paths = save_evaluation(examples, metrics, args.output_dir)
    print(json.dumps({"metrics": metrics, "files": paths}, ensure_ascii=False, indent=2))
