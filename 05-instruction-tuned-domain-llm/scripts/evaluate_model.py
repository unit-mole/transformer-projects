#!/usr/bin/env python
"""Evaluate either the configured LoRA adapter or the base FLAN-T5 model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.model_evaluation import evaluate_model
from src.model_loader import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-data", default=str(PROJECT_DIR / "data/evaluation_prompts.jsonl"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs"))
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    model = load_model(force_base=args.base_only)
    summary = evaluate_model(model, args.evaluation_data, args.output_dir, args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
