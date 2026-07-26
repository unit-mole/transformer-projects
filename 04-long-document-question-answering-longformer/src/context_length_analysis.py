from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


BUCKET_ORDER = ["0-512", "513-1024", "1025-2048", "2049-4096", "4097+"]


def aggregate_context_length_metrics(results: pd.DataFrame) -> pd.DataFrame:
    required = {
        "context_length_bucket",
        "exact_match",
        "token_f1",
        "evidence_recall",
        "latency_seconds",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Context analysis is missing columns: {sorted(missing)}")

    if results.empty:
        return pd.DataFrame(
            columns=[
                "context_length_bucket",
                "examples",
                "exact_match",
                "token_f1",
                "evidence_recall",
                "average_latency_seconds",
            ]
        )

    summary = (
        results.groupby("context_length_bucket", observed=False)
        .agg(
            examples=("example_id", "count"),
            exact_match=("exact_match", "mean"),
            token_f1=("token_f1", "mean"),
            evidence_recall=("evidence_recall", "mean"),
            average_latency_seconds=("latency_seconds", "mean"),
        )
        .reset_index()
    )
    rank = {name: index for index, name in enumerate(BUCKET_ORDER)}
    summary["_rank"] = summary["context_length_bucket"].map(rank).fillna(999)
    return summary.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)


def save_context_analysis(
    summary: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_directory / "context_length_analysis.csv", index=False)
    records = summary.to_dict(orient="records")
    (output_directory / "context_length_analysis.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
