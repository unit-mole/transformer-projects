from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .translation_pipeline import TranslationPipeline, build_default_pipeline


def _metrics(predictions: list[str], references: list[str]) -> dict[str, Any]:
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError(
            "sacrebleu is required for evaluation. Install requirements.txt."
        ) from exc

    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="none")
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return {
        "sacrebleu": round(float(bleu.score), 4),
        "sacrebleu_signature": str(bleu),
        "chrf": round(float(chrf.score), 4),
    }


def evaluate_direction(
    dataframe: pd.DataFrame,
    *,
    source_column: str,
    target_column: str,
    direction: str,
    pipeline: TranslationPipeline | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    translator = pipeline or build_default_pipeline()
    rows: list[dict[str, Any]] = []

    for source, reference in zip(
        dataframe[source_column].astype(str),
        dataframe[target_column].astype(str),
    ):
        start = time.perf_counter()
        result = translator.translate(source, direction=direction)
        wall_latency = time.perf_counter() - start
        rows.append(
            {
                "direction": direction,
                "source_text": source,
                "reference_translation": reference,
                "predicted_translation": result.translated_text,
                "confidence_proxy": result.confidence_score,
                "model_latency_seconds": result.latency_seconds,
                "wall_latency_seconds": wall_latency,
                "model_id": result.model_id,
            }
        )

    examples = pd.DataFrame(rows)
    metric_values = _metrics(
        examples["predicted_translation"].tolist(),
        examples["reference_translation"].tolist(),
    )
    latency = examples["wall_latency_seconds"]
    summary = {
        "direction": direction,
        "examples": int(len(examples)),
        **metric_values,
        "average_latency_seconds": round(float(latency.mean()), 6),
        "minimum_latency_seconds": round(float(latency.min()), 6),
        "maximum_latency_seconds": round(float(latency.max()), 6),
    }
    return examples, summary


def evaluate_bidirectional(
    dataframe: pd.DataFrame,
    *,
    english_column: str = "english",
    hindi_column: str = "hindi",
    pipeline: TranslationPipeline | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    translator = pipeline or build_default_pipeline()
    en_hi_examples, en_hi_summary = evaluate_direction(
        dataframe,
        source_column=english_column,
        target_column=hindi_column,
        direction="en_hi",
        pipeline=translator,
    )
    hi_en_examples, hi_en_summary = evaluate_direction(
        dataframe,
        source_column=hindi_column,
        target_column=english_column,
        direction="hi_en",
        pipeline=translator,
    )
    examples = pd.concat([en_hi_examples, hi_en_examples], ignore_index=True)
    return examples, {"en_hi": en_hi_summary, "hi_en": hi_en_summary}


def save_evaluation(
    examples: pd.DataFrame,
    metrics: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    examples_path = destination / "translation_examples.csv"
    metrics_path = destination / "model_metrics.json"
    examples.to_csv(examples_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "examples": str(examples_path),
        "metrics": str(metrics_path),
    }
