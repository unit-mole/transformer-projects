#!/usr/bin/env python
"""Build, validate, save, and visualize the public self-authored dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import save_jsonl, validate_and_clean_records
from src.instruction_dataset_builder import build_dataset
from src.visualization import create_dataset_visualizations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "ml_ds_instruction_dataset.jsonl"))
    parser.add_argument("--report", default=str(PROJECT_ROOT / "outputs" / "dataset_validation_report.json"))
    args = parser.parse_args()

    records = build_dataset()
    cleaned, report = validate_and_clean_records(records)
    save_jsonl(cleaned, args.output)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    create_dataset_visualizations(cleaned, PROJECT_ROOT / "outputs")
    print(json.dumps({"dataset": args.output, "records": len(cleaned), "validation": report.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
