from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .text_preprocessing import clean_parallel_pair


ENGLISH_ALIASES = [
    "english",
    "en",
    "sentence_en",
    "source_en",
    "english_text",
]
HINDI_ALIASES = [
    "hindi",
    "hi",
    "sentence_hi",
    "target_hi",
    "hindi_text",
]


@dataclass(frozen=True)
class ParallelColumns:
    english: str
    hindi: str


def _parse_translation_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def identify_parallel_columns(dataframe: pd.DataFrame) -> ParallelColumns:
    lower_map = {str(column).lower(): str(column) for column in dataframe.columns}
    english = next((lower_map[name] for name in ENGLISH_ALIASES if name in lower_map), None)
    hindi = next((lower_map[name] for name in HINDI_ALIASES if name in lower_map), None)
    if english and hindi:
        return ParallelColumns(english, hindi)

    if "translation" in lower_map:
        return ParallelColumns("__translation_en__", "__translation_hi__")

    if {"source", "target"}.issubset(lower_map):
        return ParallelColumns(lower_map["source"], lower_map["target"])
    if {"source_text", "target_text"}.issubset(lower_map):
        return ParallelColumns(lower_map["source_text"], lower_map["target_text"])

    raise ValueError(
        "Could not identify English and Hindi columns. "
        f"Available columns: {list(dataframe.columns)}"
    )


def preprocess_parallel_dataframe(
    dataframe: pd.DataFrame,
    *,
    columns: ParallelColumns | None = None,
    max_characters: int = 5000,
) -> pd.DataFrame:
    selected = columns or identify_parallel_columns(dataframe)

    if selected.english.startswith("__translation"):
        translation_column = next(
            column for column in dataframe.columns if str(column).lower() == "translation"
        )
        parsed = dataframe[translation_column].map(_parse_translation_value)
        english_values = parsed.map(lambda item: item.get("en", "") if item else "")
        hindi_values = parsed.map(lambda item: item.get("hi", "") if item else "")
    else:
        english_values = dataframe[selected.english]
        hindi_values = dataframe[selected.hindi]

    rows = []
    for english, hindi in zip(english_values, hindi_values):
        try:
            clean_en, clean_hi = clean_parallel_pair(
                english,
                hindi,
                max_characters=max_characters,
            )
        except ValueError:
            continue
        if clean_en and clean_hi:
            rows.append({"english": clean_en, "hindi": clean_hi})

    cleaned = pd.DataFrame(rows)
    if cleaned.empty:
        return cleaned
    return cleaned.drop_duplicates(["english", "hindi"]).reset_index(drop=True)


def deterministic_split(
    dataframe: pd.DataFrame,
    *,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below 1.")

    shuffled = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_end = int(len(shuffled) * train_ratio)
    validation_end = train_end + int(len(shuffled) * validation_ratio)
    return {
        "train": shuffled.iloc[:train_end].reset_index(drop=True),
        "validation": shuffled.iloc[train_end:validation_end].reset_index(drop=True),
        "test": shuffled.iloc[validation_end:].reset_index(drop=True),
    }
