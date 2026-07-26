#!/usr/bin/env python
"""Validate the dataset and export reproducible statistics and charts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.data_preprocessing import dataset_statistics, load_jsonl, validate_records
from src.visualization import create_dataset_charts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(PROJECT_DIR / "data/ml_ds_instruction_dataset.jsonl"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs"))
    args = parser.parse_args()
    records = load_jsonl(args.dataset)
    report = validate_records(records)
    stats = dataset_statistics(records)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    create_dataset_charts(args.dataset, output)
    print(json.dumps({"validation": report, "statistics": stats}, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
