from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data_preprocessing import (
    DataValidationReport,
    prepare_documents,
    prepare_qrels,
    prepare_queries,
)


@dataclass
class RankingDataset:
    documents: pd.DataFrame
    queries: pd.DataFrame
    qrels: pd.DataFrame
    document_report: DataValidationReport
    query_report: DataValidationReport

    @property
    def relevance_lookup(self) -> dict[str, dict[str, float]]:
        lookup: dict[str, dict[str, float]] = {}
        for row in self.qrels.itertuples(index=False):
            lookup.setdefault(str(row.query_id), {})[str(row.document_id)] = float(
                row.relevance
            )
        return lookup


def load_ranking_dataset(
    documents_path: str | Path,
    queries_path: str | Path,
    qrels_path: str | Path,
) -> RankingDataset:
    documents_raw = pd.read_csv(documents_path)
    queries_raw = pd.read_csv(queries_path)
    qrels_raw = pd.read_csv(qrels_path)

    documents, document_report = prepare_documents(documents_raw)
    queries, query_report = prepare_queries(queries_raw)
    qrels = prepare_qrels(
        qrels_raw,
        valid_query_ids=queries["query_id"],
        valid_document_ids=documents["document_id"],
    )

    if documents.empty:
        raise ValueError("No valid documents remain after preprocessing.")
    if queries.empty:
        raise ValueError("No valid queries remain after preprocessing.")
    if qrels.empty:
        raise ValueError("No valid relevance labels remain after preprocessing.")

    return RankingDataset(
        documents=documents,
        queries=queries,
        qrels=qrels,
        document_report=document_report,
        query_report=query_report,
    )
