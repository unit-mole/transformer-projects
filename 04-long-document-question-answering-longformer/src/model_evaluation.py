from __future__ import annotations

import json
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from .document_chunking import context_length_bucket
from .text_preprocessing import normalize_text


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: object) -> str:
    value = normalize_text(text, preserve_paragraphs=False).lower()
    value = value.translate(_PUNCT_TABLE)
    value = _ARTICLES.sub(" ", value)
    return " ".join(value.split())


def exact_match(prediction: object, reference: object) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: object, reference: object) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    overlap = Counter(predicted_tokens) & Counter(reference_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(predicted_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def evidence_recall(
    predicted_evidence: object,
    reference_evidence: object,
    minimum_overlap: float = 0.50,
) -> float:
    predicted = normalize_answer(predicted_evidence)
    reference = normalize_answer(reference_evidence)
    if not predicted or not reference:
        return 0.0
    if reference in predicted or predicted in reference:
        return 1.0
    predicted_tokens = set(predicted.split())
    reference_tokens = set(reference.split())
    if not reference_tokens:
        return 0.0
    overlap = len(predicted_tokens & reference_tokens) / len(reference_tokens)
    return float(overlap >= minimum_overlap)


def qualitative_observation(
    em: float,
    f1: float,
    evidence: float,
    confidence_proxy: float,
    warning_text: str,
) -> str:
    if em == 1.0 and evidence == 1.0:
        return "Exact answer and supporting evidence recovered."
    if f1 >= 0.70 and evidence == 1.0:
        return "Answer is partially different but grounded in the reference evidence."
    if evidence == 0.0 and f1 > 0.0:
        return "Answer overlaps the reference, but the supporting paragraph was not recovered."
    if confidence_proxy < 0.01:
        return "Very low confidence proxy; manual review is required."
    if "window" in warning_text.lower():
        return "Long-context windowing was used; inspect possible boundary effects."
    return "Incorrect or weak answer; inspect span selection and question ambiguity."


def evaluate_dataframe(
    pipeline: Any,
    frame: pd.DataFrame,
    max_length: Optional[int] = None,
    stride: Optional[int] = None,
) -> pd.DataFrame:
    required = {
        "example_id",
        "question",
        "answer",
        "reference_evidence",
        "document",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Evaluation frame is missing: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for _, item in frame.iterrows():
        started = time.perf_counter()
        try:
            result = pipeline.answer(
                question=item["question"],
                document_text=item["document"],
                source_name=item.get("document_name", item["example_id"]),
                max_length=max_length,
                stride=stride,
            )
            result_dict = result.to_dict()
            error = ""
        except Exception as exc:
            result_dict = {
                "answer": "",
                "supporting_paragraph": "",
                "confidence_proxy": 0.0,
                "document_token_count": None,
                "window_count": 0,
                "latency_seconds": time.perf_counter() - started,
                "warnings": [],
            }
            error = f"{type(exc).__name__}: {exc}"

        em = exact_match(result_dict["answer"], item["answer"])
        f1 = token_f1(result_dict["answer"], item["answer"])
        ev_recall = evidence_recall(
            result_dict["supporting_paragraph"],
            item["reference_evidence"],
        )
        warnings = " | ".join(result_dict.get("warnings", []))
        token_count = result_dict.get("document_token_count")
        if token_count is None:
            token_count = len(str(item["document"]).split())

        rows.append(
            {
                "example_id": item["example_id"],
                "source_type": item.get("source_type", "unknown"),
                "document_name": item.get("document_name", ""),
                "question": item["question"],
                "reference_answer": item["answer"],
                "predicted_answer": result_dict["answer"],
                "reference_evidence": item["reference_evidence"],
                "predicted_evidence": result_dict["supporting_paragraph"],
                "exact_match": em,
                "token_f1": f1,
                "evidence_recall": ev_recall,
                "confidence_proxy": result_dict["confidence_proxy"],
                "document_token_count": int(token_count),
                "context_length_bucket": context_length_bucket(int(token_count)),
                "window_count": result_dict["window_count"],
                "latency_seconds": result_dict["latency_seconds"],
                "warnings": warnings,
                "error": error,
                "observation": qualitative_observation(
                    em,
                    f1,
                    ev_recall,
                    float(result_dict["confidence_proxy"]),
                    warnings,
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_evaluation(results: pd.DataFrame) -> dict[str, Any]:
    if results.empty:
        return {
            "status": "not_run",
            "examples": 0,
            "exact_match": None,
            "token_f1": None,
            "evidence_recall": None,
            "average_latency_seconds": None,
        }
    valid = results[results["error"] == ""] if "error" in results else results
    if valid.empty:
        return {
            "status": "failed",
            "examples": int(len(results)),
            "exact_match": None,
            "token_f1": None,
            "evidence_recall": None,
            "average_latency_seconds": None,
        }
    return {
        "status": "completed",
        "examples": int(len(valid)),
        "exact_match": float(valid["exact_match"].mean()),
        "token_f1": float(valid["token_f1"].mean()),
        "evidence_recall": float(valid["evidence_recall"].mean()),
        "average_latency_seconds": float(valid["latency_seconds"].mean()),
        "average_confidence_proxy": float(valid["confidence_proxy"].mean()),
    }


def save_evaluation_outputs(
    results: pd.DataFrame,
    output_directory: str | Path,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_directory / "qa_examples.csv", index=False)

    summary = summarize_evaluation(results)
    (output_directory / "model_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def write_manual_error_analysis(
    results: pd.DataFrame,
    output_path: str | Path,
    maximum_examples: int = 12,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if results.empty:
        text = (
            "# Manual Error Analysis\n\n"
            "Evaluation has not been run. Execute `python scripts/evaluate_model.py` "
            "and review the generated examples before making model-performance claims.\n"
        )
        output_path.write_text(text, encoding="utf-8")
        return output_path

    ordered = results.sort_values(
        ["evidence_recall", "token_f1", "confidence_proxy"],
        ascending=[True, True, True],
    ).head(maximum_examples)

    lines = [
        "# Manual Error Analysis",
        "",
        "The examples below are generated from actual evaluation outputs. Review each "
        "prediction before publishing conclusions.",
        "",
    ]
    for _, row in ordered.iterrows():
        lines.extend(
            [
                f"## {row['example_id']}",
                "",
                f"- **Question:** {row['question']}",
                f"- **Reference answer:** {row['reference_answer']}",
                f"- **Predicted answer:** {row['predicted_answer']}",
                f"- **Exact Match:** {row['exact_match']:.3f}",
                f"- **Token F1:** {row['token_f1']:.3f}",
                f"- **Evidence recall:** {row['evidence_recall']:.3f}",
                f"- **Confidence proxy:** {row['confidence_proxy']:.6f}",
                f"- **Observation:** {row['observation']}",
                f"- **Warnings/error:** {row['warnings'] or row['error'] or 'None'}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
