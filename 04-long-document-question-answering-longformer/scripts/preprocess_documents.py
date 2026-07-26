from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import InferenceConfig
from src.data_preprocessing import (
    attach_document_text,
    load_qa_pairs,
    save_preprocessed_dataset,
)


def main() -> None:
    config = InferenceConfig().validate()
    qa_path = PROJECT_ROOT / "data" / "sample_qa_pairs.csv"
    output_path = PROJECT_ROOT / "outputs" / "preprocessed_sample_qa.csv"

    frame = load_qa_pairs(qa_path)
    frame = attach_document_text(frame, config.sample_directory)
    save_preprocessed_dataset(frame, output_path)
    print(f"Saved {len(frame)} preprocessed examples to {output_path}")


if __name__ == "__main__":
    main()
