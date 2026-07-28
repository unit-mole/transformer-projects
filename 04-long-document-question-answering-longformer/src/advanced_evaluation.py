from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
CONTEXT_BUCKETS = ["0-512", "513-1024", "1025-2048", "2049-4096", "4097+"]


def normalize_answer(text: Any) -> str:
    value = " ".join(str(text or "").strip().lower().split())
    value = value.translate(_PUNCT_TABLE)
    value = _ARTICLES.sub(" ", value)
    return " ".join(value.split())


def exact_match(prediction: Any, reference: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: Any, reference: Any) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    common = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(predicted_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_references(
    prediction: Any,
    references: Iterable[Any],
    metric: Any,
) -> float:
    values = [metric(prediction, reference) for reference in references]
    return max(values) if values else 0.0


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [text]


def evidence_token_recall(predicted_evidence: Any, references: Iterable[Any]) -> float:
    predicted = set(normalize_answer(predicted_evidence).split())
    if not predicted:
        return 0.0
    scores: list[float] = []
    for reference in references:
        reference_tokens = set(normalize_answer(reference).split())
        if reference_tokens:
            scores.append(len(predicted & reference_tokens) / len(reference_tokens))
    return max(scores) if scores else 0.0


def evidence_recovered(
    predicted_evidence: Any,
    references: Iterable[Any],
    threshold: float = 0.50,
) -> float:
    predicted = normalize_answer(predicted_evidence)
    if not predicted:
        return 0.0
    for reference in references:
        normalized_reference = normalize_answer(reference)
        if not normalized_reference:
            continue
        if normalized_reference in predicted or predicted in normalized_reference:
            return 1.0
    return float(evidence_token_recall(predicted_evidence, references) >= threshold)


def context_length_bucket(token_count: int) -> str:
    if token_count <= 512:
        return "0-512"
    if token_count <= 1024:
        return "513-1024"
    if token_count <= 2048:
        return "1025-2048"
    if token_count <= 4096:
        return "2049-4096"
    return "4097+"


def answer_position_bucket(answer_token_position: int | None) -> str:
    if answer_token_position is None:
        return "unknown"
    try:
        if np.isnan(answer_token_position):
            return "unknown"
    except TypeError:
        pass
    if int(answer_token_position) < 0:
        return "unknown"
    answer_token_position = int(answer_token_position)
    if answer_token_position <= 512:
        return "within-first-512"
    if answer_token_position <= 1024:
        return "513-1024"
    if answer_token_position <= 2048:
        return "1025-2048"
    if answer_token_position <= 4096:
        return "2049-4096"
    return "4097+"


def classify_error(em: float, f1: float, evidence: float, answer_position: str) -> str:
    if em == 1.0 and evidence == 1.0:
        return "exact-and-grounded"
    if f1 >= 0.70 and evidence == 1.0:
        return "partial-but-grounded"
    if f1 > 0 and evidence == 0:
        return "answer-overlap-wrong-evidence"
    if f1 == 0 and evidence == 1:
        return "wrong-span-correct-evidence"
    if answer_position != "within-first-512" and answer_position != "unknown":
        return "long-context-failure"
    return "incorrect-answer"


def score_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    references = _parse_list(row.get("reference_answers_json"))
    evidence_references = _parse_list(row.get("reference_evidence_json"))
    prediction = row.get("predicted_answer", "")
    predicted_evidence = row.get("predicted_evidence", "")

    em = metric_max_over_references(prediction, references, exact_match)
    f1 = metric_max_over_references(prediction, references, token_f1)
    evidence_continuous = evidence_token_recall(predicted_evidence, evidence_references)
    evidence_binary = evidence_recovered(predicted_evidence, evidence_references)
    answer_position = answer_position_bucket(row.get("answer_token_position"))

    scored = dict(row)
    token_count_value = row.get("document_token_count")
    try:
        token_count = 0 if token_count_value is None or np.isnan(token_count_value) else int(token_count_value)
    except TypeError:
        token_count = int(token_count_value or 0)
    answer_position_value = row.get("answer_token_position")
    try:
        answer_beyond = bool(
            answer_position_value is not None
            and not np.isnan(answer_position_value)
            and int(answer_position_value) > 512
        )
    except TypeError:
        answer_beyond = bool(answer_position_value is not None and int(answer_position_value) > 512)

    scored.update(
        {
            "exact_match": float(em),
            "token_f1": float(f1),
            "evidence_token_recall": float(evidence_continuous),
            "evidence_recovered": float(evidence_binary),
            "context_length_bucket": context_length_bucket(token_count),
            "answer_position_bucket": answer_position,
            "answer_beyond_512": answer_beyond,
            "error_category": classify_error(em, f1, evidence_binary, answer_position),
        }
    )
    return scored


def score_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return pd.DataFrame(score_prediction_row(row) for row in frame.to_dict(orient="records"))


def summarize_results(scored: pd.DataFrame, model_name: str) -> dict[str, Any]:
    if scored.empty:
        return {
            "status": "not_run",
            "model_name": model_name,
            "examples": 0,
            "exact_match": None,
            "token_f1": None,
            "evidence_recovered": None,
            "evidence_token_recall": None,
            "average_latency_seconds": None,
            "throughput_examples_per_second": None,
        }
    valid = scored[scored.get("error", "") == ""] if "error" in scored else scored
    if valid.empty:
        return {"status": "failed", "model_name": model_name, "examples": 0}
    average_latency = float(valid["latency_seconds"].mean())
    return {
        "status": "completed",
        "model_name": model_name,
        "examples": int(len(valid)),
        "exact_match": float(valid["exact_match"].mean()),
        "token_f1": float(valid["token_f1"].mean()),
        "evidence_recovered": float(valid["evidence_recovered"].mean()),
        "evidence_token_recall": float(valid["evidence_token_recall"].mean()),
        "average_latency_seconds": average_latency,
        "median_latency_seconds": float(valid["latency_seconds"].median()),
        "p95_latency_seconds": float(valid["latency_seconds"].quantile(0.95)),
        "throughput_examples_per_second": 1.0 / average_latency if average_latency > 0 else None,
        "average_confidence_proxy": float(valid["confidence_proxy"].mean()),
        "average_window_count": float(valid["window_count"].mean()),
        "peak_gpu_memory_mb": float(valid["peak_gpu_memory_mb"].max())
        if "peak_gpu_memory_mb" in valid and valid["peak_gpu_memory_mb"].notna().any()
        else None,
        "answers_beyond_512_examples": int(valid["answer_beyond_512"].sum()),
    }


def aggregate_by_column(scored: pd.DataFrame, column: str) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    summary = (
        scored.groupby(column, observed=False)
        .agg(
            examples=("example_id", "count"),
            exact_match=("exact_match", "mean"),
            token_f1=("token_f1", "mean"),
            evidence_recovered=("evidence_recovered", "mean"),
            evidence_token_recall=("evidence_token_recall", "mean"),
            average_latency_seconds=("latency_seconds", "mean"),
            average_windows=("window_count", "mean"),
        )
        .reset_index()
    )
    if column == "context_length_bucket":
        order = {name: index for index, name in enumerate(CONTEXT_BUCKETS)}
        summary["_order"] = summary[column].map(order).fillna(999)
        summary = summary.sort_values("_order").drop(columns="_order")
    return summary.reset_index(drop=True)


def confidence_analysis(scored: pd.DataFrame) -> dict[str, Any]:
    if scored.empty:
        return {"status": "not_run"}
    confidence = scored["confidence_proxy"].astype(float)
    f1 = scored["token_f1"].astype(float)
    exact = scored["exact_match"].astype(float)
    try:
        from scipy.stats import spearmanr

        correlation = float(spearmanr(confidence, f1, nan_policy="omit").statistic)
    except Exception:
        correlation = float(pd.Series(confidence).corr(pd.Series(f1), method="spearman"))
    return {
        "status": "completed",
        "examples": int(len(scored)),
        "spearman_confidence_vs_f1": correlation,
        "mean_confidence_exact": float(confidence[exact == 1].mean())
        if (exact == 1).any()
        else None,
        "mean_confidence_incorrect": float(confidence[exact == 0].mean())
        if (exact == 0).any()
        else None,
        "note": (
            "Confidence is an uncalibrated start/end probability proxy. The analysis "
            "measures association with correctness, not probability calibration."
        ),
    }


def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path
