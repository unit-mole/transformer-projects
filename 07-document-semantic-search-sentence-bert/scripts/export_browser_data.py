#!/usr/bin/env python
"""Export processed corpus artifacts and verified evaluation JSON to web/data."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.export_for_browser import export_browser_data

EVALUATION_FILES = (
    "model_metrics.json",
    "recall_at_k_results.json",
    "mrr_results.json",
    "query_latency_results.json",
    "cosine_similarity_analysis.json",
)


def copy_evaluation_artifacts(outputs_dir: Path, web_data_dir: Path) -> list[Path]:
    """Copy available, valid evaluation JSON files into the static app."""
    copied: list[Path] = []
    web_data_dir.mkdir(parents=True, exist_ok=True)
    for filename in EVALUATION_FILES:
        source = outputs_dir / filename
        if not source.is_file():
            continue
        with source.open(encoding="utf-8") as handle:
            json.load(handle)
        destination = web_data_dir / filename
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--web-data-dir", type=Path, default=PROJECT_ROOT / "web/data")
    args = parser.parse_args()

    copied = export_browser_data(args.processed_dir, args.web_data_dir)
    copied.extend(copy_evaluation_artifacts(args.outputs_dir, args.web_data_dir))
    print("Exported browser artifacts:")
    for path in copied:
        print(f"- {path}")


if __name__ == "__main__":
    main()
