from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .text_preprocessing import clean_text, combine_title_and_document


@dataclass(frozen=True)
class DataValidationReport:
    input_rows: int
    output_rows: int
    removed_missing: int
    removed_duplicates: int


def prepare_documents(frame: pd.DataFrame) -> tuple[pd.DataFrame, DataValidationReport]:
    required = {"document_id", "title", "document"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Document data is missing columns: {sorted(missing)}")

    result = frame.copy()
    input_rows = len(result)
    result["document_id"] = result["document_id"].map(clean_text)
    result["title"] = result["title"].map(clean_text)
    result["document"] = result["document"].map(clean_text)

    valid_mask = (
        result["document_id"].ne("")
        & result["document"].str.len().ge(20)
    )
    removed_missing = int((~valid_mask).sum())
    result = result.loc[valid_mask].copy()

    before_deduplication = len(result)
    result = result.drop_duplicates(subset=["document_id"], keep="first")
    result = result.drop_duplicates(subset=["document"], keep="first")
    removed_duplicates = before_deduplication - len(result)

    result["search_text"] = [
        combine_title_and_document(title, document)
        for title, document in zip(result["title"], result["document"])
    ]
    result = result.reset_index(drop=True)

    return result, DataValidationReport(
        input_rows=input_rows,
        output_rows=len(result),
        removed_missing=removed_missing,
        removed_duplicates=removed_duplicates,
    )


def prepare_queries(frame: pd.DataFrame) -> tuple[pd.DataFrame, DataValidationReport]:
    required = {"query_id", "query"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Query data is missing columns: {sorted(missing)}")

    result = frame.copy()
    input_rows = len(result)
    result["query_id"] = result["query_id"].map(clean_text)
    result["query"] = result["query"].map(clean_text)

    valid_mask = result["query_id"].ne("") & result["query"].str.len().ge(3)
    removed_missing = int((~valid_mask).sum())
    result = result.loc[valid_mask].copy()

    before_deduplication = len(result)
    result = result.drop_duplicates(subset=["query_id"], keep="first")
    removed_duplicates = before_deduplication - len(result)
    result = result.reset_index(drop=True)

    return result, DataValidationReport(
        input_rows=input_rows,
        output_rows=len(result),
        removed_missing=removed_missing,
        removed_duplicates=removed_duplicates,
    )


def prepare_qrels(
    frame: pd.DataFrame,
    valid_query_ids: Iterable[str],
    valid_document_ids: Iterable[str],
) -> pd.DataFrame:
    required = {"query_id", "document_id", "relevance"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Qrels data is missing columns: {sorted(missing)}")

    result = frame.copy()
    result["query_id"] = result["query_id"].map(clean_text)
    result["document_id"] = result["document_id"].map(clean_text)
    result["relevance"] = pd.to_numeric(result["relevance"], errors="coerce")

    result = result.dropna(subset=["relevance"])
    result = result[result["relevance"].between(0, 3)]
    result = result[result["query_id"].isin(set(valid_query_ids))]
    result = result[result["document_id"].isin(set(valid_document_ids))]
    result = result.drop_duplicates(
        subset=["query_id", "document_id"],
        keep="last",
    )
    return result.reset_index(drop=True)
