from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .text_preprocessing import normalize_text


REQUIRED_QA_COLUMNS = {
    "example_id",
    "document_name",
    "question",
    "answer",
    "reference_evidence",
}


def load_qa_pairs(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_QA_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing QA columns: {sorted(missing)}")

    cleaned = frame.copy()
    for column in ["question", "answer", "reference_evidence", "document_name"]:
        cleaned[column] = cleaned[column].map(
            lambda value: normalize_text(value, preserve_paragraphs=False)
        )
    cleaned = cleaned[
        (cleaned["question"] != "")
        & (cleaned["answer"] != "")
        & (cleaned["document_name"] != "")
    ].reset_index(drop=True)
    return cleaned


def attach_document_text(
    frame: pd.DataFrame,
    sample_directory: str | Path,
) -> pd.DataFrame:
    directory = Path(sample_directory)
    output = frame.copy()
    texts = []
    for document_name in output["document_name"]:
        path = directory / Path(document_name).name
        if not path.exists():
            raise FileNotFoundError(f"Sample document not found: {path}")
        texts.append(normalize_text(path.read_text(encoding="utf-8")))
    output["document"] = texts
    output["document_character_count"] = output["document"].str.len()
    output["document_word_count"] = output["document"].str.split().str.len()
    return output


def save_preprocessed_dataset(frame: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
