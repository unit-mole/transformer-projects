from __future__ import annotations

import json
import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

_ARTICLES = {"a", "an", "the"}

def normalize_answer(value: object) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"[^\w\s'-]", " ", text)
    tokens = [token for token in text.split() if token not in _ARTICLES]
    return " ".join(tokens)

def parse_reference_answers(value: object) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]

def exact_match_score(prediction: str, references: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(ref) for ref in references))

def vqa_consensus_score(prediction: str, references: Sequence[str]) -> float:
    """Approximate standard VQA consensus: min(matches / 3, 1)."""
    pred = normalize_answer(prediction)
    matches = sum(pred == normalize_answer(ref) for ref in references)
    return min(matches / 3.0, 1.0)

def evaluate_records(records: Iterable[Mapping[str, object]]) -> dict:
    rows = []
    for record in records:
        prediction = str(record["prediction"])
        references = parse_reference_answers(record.get("answers", record.get("answer", "")))
        rows.append({
            "question_type": str(record.get("question_type", "other")),
            "answer_type": str(record.get("answer_type", "other")),
            "exact_match": exact_match_score(prediction, references),
            "vqa_score": vqa_consensus_score(prediction, references),
        })
    if not rows:
        return {"count": 0, "exact_match": None, "vqa_accuracy": None, "by_question_type": {}}

    by_type = {}
    for question_type in sorted({row["question_type"] for row in rows}):
        group = [row for row in rows if row["question_type"] == question_type]
        by_type[question_type] = {
            "count": len(group),
            "vqa_accuracy": sum(row["vqa_score"] for row in group) / len(group),
        }

    return {
        "count": len(rows),
        "exact_match": sum(row["exact_match"] for row in rows) / len(rows),
        "vqa_accuracy": sum(row["vqa_score"] for row in rows) / len(rows),
        "by_question_type": by_type,
    }
