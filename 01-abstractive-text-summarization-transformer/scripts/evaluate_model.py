from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import prepare_dataframe
from src.model_evaluation import evaluate_dataframe, save_evaluation_outputs
from src.summarization_model import GenerationSettings, TransformerSummarizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the DistilBART summarizer.")
    parser.add_argument("--input-csv", default="data/sample_summaries.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compute-bertscore", action="store_true")
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=120)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=2.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = PROJECT_ROOT / args.input_csv
    frame = prepare_dataframe(pd.read_csv(input_path))
    settings = GenerationSettings(
        min_length=args.min_length,
        max_length=args.max_length,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "runs" / timestamp
    results, metrics = evaluate_dataframe(
        frame,
        TransformerSummarizer(),
        settings,
        compute_bert_score=args.compute_bertscore,
        limit=args.limit,
    )
    save_evaluation_outputs(results, metrics, output_dir)
    print(f"Saved evaluation to: {output_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
