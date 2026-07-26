from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.context_length_analysis import (
    aggregate_context_length_metrics,
    save_context_analysis,
)
from src.visualization import plot_context_length_analysis


def main() -> None:
    results_path = PROJECT_ROOT / "outputs" / "qa_examples.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            "Run `python scripts/evaluate_model.py` before context analysis."
        )

    results = pd.read_csv(results_path)
    if results.empty:
        raise ValueError("The evaluation results file is empty.")

    summary = aggregate_context_length_metrics(results)
    save_context_analysis(summary, PROJECT_ROOT / "outputs")
    plot_context_length_analysis(
        summary,
        PROJECT_ROOT / "outputs" / "context_length_analysis.png",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
