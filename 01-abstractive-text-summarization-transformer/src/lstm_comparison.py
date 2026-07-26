from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .model_evaluation import compute_bertscore, compute_rouge
from .text_preprocessing import word_count

REQUIRED_COLUMNS = {"id", "article", "reference_summary", "lstm_summary"}


def load_lstm_predictions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"LSTM comparison CSV is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["article", "reference_summary", "lstm_summary"]).copy()
    placeholder = frame["lstm_summary"].astype(str).str.contains("Paste the actual", case=False)
    frame = frame[~placeholder].reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            "No actual LSTM predictions were found. Replace the template text with real outputs."
        )
    return frame


def metric_row(
    name: str,
    predictions: list[str],
    references: list[str],
    latencies: list[float] | None = None,
    *,
    include_bertscore: bool = False,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"model": name, **compute_rouge(predictions, references)}
    if include_bertscore:
        metrics.update(compute_bertscore(predictions, references))
    metrics["average_inference_seconds"] = (
        sum(latencies) / len(latencies) if latencies else None
    )
    metrics["average_summary_words"] = (
        sum(word_count(item) for item in predictions) / max(len(predictions), 1)
    )
    return metrics


def build_comparison(
    frame: pd.DataFrame,
    *,
    transformer_column: str = "transformer_summary",
    transformer_latency_column: str = "inference_seconds",
    include_bertscore: bool = False,
) -> pd.DataFrame:
    references = frame["reference_summary"].astype(str).tolist()
    rows = [
        metric_row(
            "LSTM Seq2Seq with Attention",
            frame["lstm_summary"].astype(str).tolist(),
            references,
            frame["lstm_inference_seconds"].tolist()
            if "lstm_inference_seconds" in frame.columns
            else None,
            include_bertscore=include_bertscore,
        )
    ]
    if transformer_column in frame.columns:
        rows.append(
            metric_row(
                "DistilBART Transformer",
                frame[transformer_column].astype(str).tolist(),
                references,
                frame[transformer_latency_column].tolist()
                if transformer_latency_column in frame.columns
                else None,
                include_bertscore=include_bertscore,
            )
        )
    return pd.DataFrame(rows)
