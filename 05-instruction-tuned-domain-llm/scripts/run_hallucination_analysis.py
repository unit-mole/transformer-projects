#!/usr/bin/env python
"""Apply semi-automated risk flags to generated-response outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.hallucination_analysis import flag_hallucination_risks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PROJECT_DIR / "outputs/generated_response_examples.csv"))
    parser.add_argument("--output", default=str(PROJECT_DIR / "outputs/hallucination_review.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    reviews = [flag_hallucination_risks(row.get("instruction", ""), row.get("generated_answer", "")) for _, row in frame.iterrows()]
    review_frame = pd.DataFrame(reviews)
    combined = pd.concat([frame.reset_index(drop=True), review_frame], axis=1)
    combined.to_csv(args.output, index=False)
    print(f"Saved {len(combined)} rows to {args.output}")


if __name__ == "__main__":
    main()
