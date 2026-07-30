#!/usr/bin/env python
"""Run model evaluation without training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation import evaluate_from_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=str(PROJECT_ROOT / "data" / "evaluation_prompts.jsonl"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bertscore", action="store_true")
    args = parser.parse_args()
    summary = evaluate_from_file(args.prompts, args.output_dir, limit=args.limit, include_bertscore=args.bertscore)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
