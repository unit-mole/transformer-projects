#!/usr/bin/env python
"""Create a hallucination-review report from evaluation CSV output."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hallucination_analysis import analyze_hallucination_risk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-csv", default=str(PROJECT_ROOT / "outputs" / "manual_review_results.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "hallucination_analysis.md"))
    args = parser.parse_args()

    source = Path(args.evaluation_csv)
    if not source.exists():
        raise FileNotFoundError("Run scripts/evaluate_model.py first.")
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    lines = ["# Hallucination Analysis", "", "> Semi-automated flags require human factual review.", ""]
    for index, row in enumerate(rows, start=1):
        result = analyze_hallucination_risk(row.get("prompt", ""), row.get("generated_answer", ""), row.get("reference_answer", ""))
        lines.extend([
            f"## Example {index}",
            f"- **Prompt:** {row.get('prompt', '')}",
            f"- **Generated response:** {row.get('generated_answer', '')}",
            f"- **Issue types:** {', '.join(result['issue_types']) or 'None flagged'}",
            f"- **Severity:** {result['severity']}",
            "- **Manual note:** ",
            "- **Corrected explanation:** ",
            "",
        ])
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
