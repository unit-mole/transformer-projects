from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .translation_pipeline import TranslationPipeline, build_default_pipeline


OUTPUT_COLUMNS = [
    "original_text",
    "detected_language",
    "translation_direction",
    "translated_text",
    "confidence_score",
    "confidence_method",
    "latency_seconds",
    "status",
    "error",
]


def translate_dataframe(
    dataframe: pd.DataFrame,
    *,
    text_column: str,
    direction: str = "auto",
    max_rows: int = 100,
    pipeline: TranslationPipeline | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if text_column not in dataframe.columns:
        raise ValueError(
            f"Column '{text_column}' was not found. Available columns: "
            f"{list(dataframe.columns)}"
        )
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1.")

    translator = pipeline or build_default_pipeline()
    subset = dataframe.head(max_rows).copy()
    records: list[dict[str, Any]] = []

    for value in subset[text_column].tolist():
        original = "" if pd.isna(value) else str(value)
        try:
            result = translator.translate(original, direction=direction)
            records.append(
                {
                    "original_text": result.original_text,
                    "detected_language": result.detected_language,
                    "translation_direction": result.direction_label,
                    "translated_text": result.translated_text,
                    "confidence_score": result.confidence_score,
                    "confidence_method": result.confidence_method,
                    "latency_seconds": result.latency_seconds,
                    "status": "success",
                    "error": "",
                }
            )
        except Exception as exc:
            records.append(
                {
                    "original_text": original,
                    "detected_language": "",
                    "translation_direction": direction,
                    "translated_text": "",
                    "confidence_score": None,
                    "confidence_method": "",
                    "latency_seconds": None,
                    "status": "error",
                    "error": str(exc),
                }
            )

    output = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    successful = output[output["status"] == "success"]
    latency_series = pd.to_numeric(successful["latency_seconds"], errors="coerce")

    summary = {
        "requested_rows": int(min(len(dataframe), max_rows)),
        "successful_rows": int((output["status"] == "success").sum()),
        "failed_rows": int((output["status"] == "error").sum()),
        "average_latency_seconds": (
            round(float(latency_series.mean()), 6)
            if not latency_series.dropna().empty
            else None
        ),
    }
    return output, summary


def translate_csv(
    input_path: str | Path,
    *,
    text_column: str,
    direction: str = "auto",
    max_rows: int = 100,
    output_path: str | Path | None = None,
    pipeline: TranslationPipeline | None = None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    path = Path(input_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Batch translation accepts CSV files only.")

    dataframe = pd.read_csv(path)
    output, summary = translate_dataframe(
        dataframe,
        text_column=text_column,
        direction=direction,
        max_rows=max_rows,
        pipeline=pipeline,
    )

    if output_path is None:
        handle = tempfile.NamedTemporaryFile(
            prefix="translated_",
            suffix=".csv",
            delete=False,
        )
        handle.close()
        destination = Path(handle.name)
    else:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(destination, index=False, encoding="utf-8-sig")
    return output, str(destination), summary
