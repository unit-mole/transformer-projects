from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from .summarization_model import GenerationSettings, TransformerSummarizer
from .text_preprocessing import word_count


def _f1(overlap: int, prediction_count: int, reference_count: int) -> float:
    if prediction_count == 0 or reference_count == 0 or overlap == 0:
        return 0.0
    precision = overlap / prediction_count
    recall = overlap / reference_count
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _simple_rouge_pair(prediction: str, reference: str) -> dict[str, float]:
    import re
    from collections import Counter

    prediction_tokens = re.findall(r"\b[\w'-]+\b", prediction.lower())
    reference_tokens = re.findall(r"\b[\w'-]+\b", reference.lower())
    scores: dict[str, float] = {}
    for name, size in (("rouge1", 1), ("rouge2", 2)):
        prediction_ngrams = Counter(_ngrams(prediction_tokens, size))
        reference_ngrams = Counter(_ngrams(reference_tokens, size))
        overlap = sum((prediction_ngrams & reference_ngrams).values())
        scores[name] = _f1(overlap, sum(prediction_ngrams.values()), sum(reference_ngrams.values()))
    lcs = _lcs_length(prediction_tokens, reference_tokens)
    scores["rougeL"] = _f1(lcs, len(prediction_tokens), len(reference_tokens))
    return scores


def compute_rouge(predictions: Iterable[str], references: Iterable[str]) -> dict[str, float]:
    pairs = list(zip(predictions, references))
    if not pairs:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rows = [scorer.score(reference, prediction) for prediction, reference in pairs]
        return {
            metric: statistics.fmean(row[metric].fmeasure for row in rows)
            for metric in ("rouge1", "rouge2", "rougeL")
        }
    except ImportError:
        rows = [_simple_rouge_pair(prediction, reference) for prediction, reference in pairs]
        return {
            metric: statistics.fmean(row[metric] for row in rows)
            for metric in ("rouge1", "rouge2", "rougeL")
        }


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    *,
    language: str = "en",
) -> dict[str, float]:
    try:
        from bert_score import score
    except ImportError as exc:
        raise RuntimeError("Install bert-score to compute BERTScore.") from exc
    precision, recall, f1 = score(
        predictions,
        references,
        lang=language,
        verbose=False,
        rescale_with_baseline=True,
    )
    return {
        "bertscore_precision": float(precision.mean()),
        "bertscore_recall": float(recall.mean()),
        "bertscore_f1": float(f1.mean()),
    }


def evaluate_dataframe(
    frame: pd.DataFrame,
    summarizer: TransformerSummarizer,
    settings: GenerationSettings,
    *,
    compute_bert_score: bool = True,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame.head(limit).copy() if limit else frame.copy()
    required = {"article", "reference_summary"}
    if not required.issubset(work.columns):
        raise ValueError(f"Evaluation data must contain columns: {sorted(required)}")

    rows: list[dict[str, Any]] = []
    for index, row in work.iterrows():
        result = summarizer.summarize(str(row["article"]), settings)
        rows.append(
            {
                "id": row.get("id", index),
                "article": row["article"],
                "reference_summary": row["reference_summary"],
                "transformer_summary": result.summary,
                "inference_seconds": result.inference_seconds,
                "article_words": result.input_words,
                "reference_words": word_count(row["reference_summary"]),
                "generated_words": result.summary_words,
                "compression_ratio": result.compression_ratio,
                "chunks_processed": result.chunks_processed,
            }
        )

    results = pd.DataFrame(rows)
    predictions = results["transformer_summary"].astype(str).tolist()
    references = results["reference_summary"].astype(str).tolist()
    metrics: dict[str, Any] = compute_rouge(predictions, references)
    latencies = results["inference_seconds"].tolist()
    metrics.update(
        {
            "samples": len(results),
            "average_inference_seconds": statistics.fmean(latencies) if latencies else 0.0,
            "minimum_inference_seconds": min(latencies, default=0.0),
            "maximum_inference_seconds": max(latencies, default=0.0),
            "average_compression_ratio": float(results["compression_ratio"].mean())
            if len(results)
            else 0.0,
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": summarizer.model_name,
            "generation_settings": settings.__dict__,
        }
    )
    if compute_bert_score and predictions:
        metrics.update(compute_bertscore(predictions, references))
    return results, metrics


def save_evaluation_outputs(
    results: pd.DataFrame,
    metrics: dict[str, Any],
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination / "generated_summary_examples.csv", index=False)
    (destination / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    rouge = {key: metrics[key] for key in ("rouge1", "rouge2", "rougeL") if key in metrics}
    (destination / "rouge_scores.json").write_text(json.dumps(rouge, indent=2), encoding="utf-8")
    bert = {key: value for key, value in metrics.items() if key.startswith("bertscore_")}
    (destination / "bertscore_results.json").write_text(
        json.dumps(bert or {"status": "not_computed"}, indent=2), encoding="utf-8"
    )
    latency = {
        key: metrics[key]
        for key in (
            "average_inference_seconds",
            "minimum_inference_seconds",
            "maximum_inference_seconds",
        )
        if key in metrics
    }
    (destination / "inference_time_results.json").write_text(
        json.dumps(latency, indent=2), encoding="utf-8"
    )
