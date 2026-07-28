from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio_evaluation import load_config, summarize_manual_review  # noqa: E402


if __name__ == "__main__":
    config = load_config(PROJECT_ROOT)
    output_root = PROJECT_ROOT / config["outputs"]["root"]
    review_path = output_root / "manual_error_analysis_candidates.csv"
    if not review_path.exists():
        raise SystemExit(f"Missing manual review file: {review_path}")
    summary = summarize_manual_review(pd.read_csv(review_path))
    target = output_root / "manual_error_analysis_summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
