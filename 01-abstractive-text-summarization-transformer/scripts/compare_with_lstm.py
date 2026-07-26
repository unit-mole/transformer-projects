from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lstm_comparison import build_comparison, load_lstm_predictions
from src.summarization_model import GenerationSettings, TransformerSummarizer
from src.visualization import plot_model_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare actual LSTM and Transformer outputs.")
    parser.add_argument("--lstm-csv", required=True)
    parser.add_argument("--compute-bertscore", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_lstm_predictions(PROJECT_ROOT / args.lstm_csv)
    summarizer = TransformerSummarizer()
    settings = GenerationSettings()
    transformer_summaries = []
    transformer_latencies = []
    for article in frame["article"].astype(str):
        result = summarizer.summarize(article, settings)
        transformer_summaries.append(result.summary)
        transformer_latencies.append(result.inference_seconds)
    frame["transformer_summary"] = transformer_summaries
    frame["inference_seconds"] = transformer_latencies

    comparison = build_comparison(frame, include_bertscore=args.compute_bertscore)
    output_dir = PROJECT_ROOT / "outputs" / "runs" / time.strftime("lstm_compare_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "qualitative_comparison.csv", index=False)
    comparison.to_csv(output_dir / "transformer_vs_lstm_comparison.csv", index=False)
    plot_model_comparison(comparison, output_dir / "transformer_vs_lstm_comparison.png")
    print(comparison.to_string(index=False))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
