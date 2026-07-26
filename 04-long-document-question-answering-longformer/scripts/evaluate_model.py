from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import InferenceConfig
from src.data_preprocessing import attach_document_text, load_qa_pairs
from src.inference_pipeline import LongDocumentQAPipeline
from src.model_evaluation import (
    evaluate_dataframe,
    save_evaluation_outputs,
    write_manual_error_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Longformer QA on the safe sample dataset."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Runtime token window. Defaults to LONGDOCQA_MAX_LENGTH.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Token overlap between windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InferenceConfig().validate()
    qa_path = PROJECT_ROOT / "data" / "sample_qa_pairs.csv"
    output_dir = PROJECT_ROOT / "outputs"

    frame = load_qa_pairs(qa_path)
    frame = attach_document_text(frame, config.sample_directory)
    pipeline = LongDocumentQAPipeline(config)

    results = evaluate_dataframe(
        pipeline=pipeline,
        frame=frame,
        max_length=args.max_length,
        stride=args.stride,
    )
    summary = save_evaluation_outputs(results, output_dir)
    write_manual_error_analysis(
        results,
        output_dir / "manual_error_analysis.md",
    )

    print(json.dumps(summary, indent=2))
    failed = int((results["error"] != "").sum()) if "error" in results else 0
    if failed:
        print(f"Warning: {failed} evaluation row(s) failed. Review outputs/qa_examples.csv.")


if __name__ == "__main__":
    main()
