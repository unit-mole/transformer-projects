from __future__ import annotations

import re
from typing import Any

QUESTION_TYPE_PATTERNS = (
    ("yes_no", re.compile(r"^(is|are|was|were|do|does|did|can|could|has|have|will|would)\b", re.I)),
    ("number", re.compile(r"^(how many|what number|number of)\b", re.I)),
    ("color", re.compile(r"\b(what|which)\s+colou?r\b|\bcolou?r\b", re.I)),
    ("spatial", re.compile(r"\b(where|left|right|above|below|behind|front|next to|between)\b", re.I)),
    ("action", re.compile(r"\b(doing|happening|action|holding|playing|riding|eating)\b", re.I)),
    ("attribute", re.compile(r"\b(size|shape|kind|type|material|pattern)\b", re.I)),
    ("object", re.compile(r"\b(what object|what is|which object|identify)\b", re.I)),
)

def preprocess_question(value: Any, max_chars: int = 300) -> str:
    if value is None:
        raise ValueError("Please enter a question about the image.")
    question = re.sub(r"\s+", " ", str(value)).strip()
    if not question:
        raise ValueError("Please enter a question about the image.")
    if len(question) > max_chars:
        raise ValueError(f"Question is too long. Use at most {max_chars} characters.")
    if question[-1] not in "?!.":
        question += "?"
    return question

def classify_question_type(question: str) -> str:
    cleaned = preprocess_question(question)
    for label, pattern in QUESTION_TYPE_PATTERNS:
        if pattern.search(cleaned):
            return label
    return "other"

def classify_answer_type(answer: str) -> str:
    value = str(answer).strip().lower()
    if value in {"yes", "no"}:
        return "yes_no"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
        return "number"
    return "other"
