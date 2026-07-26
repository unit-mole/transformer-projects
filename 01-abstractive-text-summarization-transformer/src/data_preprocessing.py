from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from .text_preprocessing import clean_text, word_count

ARTICLE_CANDIDATES = ("article", "document", "text", "content", "news_article", "body")
SUMMARY_CANDIDATES = ("summary", "highlights", "target", "abstract", "reference_summary")


@dataclass(frozen=True)
class ColumnMapping:
    article_column: str
    summary_column: str


def _first_present(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def detect_columns(frame: pd.DataFrame) -> ColumnMapping:
    article = _first_present(frame.columns, ARTICLE_CANDIDATES)
    summary = _first_present(frame.columns, SUMMARY_CANDIDATES)
    if not article or not summary:
        raise ValueError(
            "Could not identify article and summary columns. "
            f"Available columns: {list(frame.columns)}"
        )
    return ColumnMapping(article, summary)


def prepare_dataframe(
    frame: pd.DataFrame,
    mapping: ColumnMapping | None = None,
    *,
    min_article_words: int = 25,
    min_summary_words: int = 5,
) -> pd.DataFrame:
    mapping = mapping or detect_columns(frame)
    prepared = frame.copy()
    prepared["article"] = prepared[mapping.article_column].map(clean_text)
    prepared["reference_summary"] = prepared[mapping.summary_column].map(clean_text)
    prepared["article_words"] = prepared["article"].map(word_count)
    prepared["summary_words"] = prepared["reference_summary"].map(word_count)
    prepared = prepared[
        (prepared["article_words"] >= min_article_words)
        & (prepared["summary_words"] >= min_summary_words)
    ]
    return prepared.drop_duplicates(subset=["article", "reference_summary"]).reset_index(drop=True)


def split_dataframe(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must be between 0 and 1.")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1.")

    train, temporary = train_test_split(
        frame, test_size=validation_size + test_size, random_state=random_state
    )
    relative_test = test_size / (validation_size + test_size)
    validation, test = train_test_split(
        temporary, test_size=relative_test, random_state=random_state
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
