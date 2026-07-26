"""Transparent heuristic rubric for instruction-adherence review."""

from __future__ import annotations

import re
from typing import Any

REFUSAL_MARKERS = ("i cannot help", "i can't help", "not able to help")
OFF_TOPIC_MARKERS = ("legal advice", "medical diagnosis", "investment advice", "immigration advice")


def score_instruction_adherence(instruction: str, response: str) -> dict[str, Any]:
    instruction = instruction.strip().lower()
    response = response.strip()
    lower = response.lower()
    answered = bool(response) and len(response.split()) >= 4
    safe_refusal = any(marker in lower for marker in OFF_TOPIC_MARKERS)
    unnecessary_refusal = any(marker in lower for marker in REFUSAL_MARKERS) and not safe_refusal
    requested_code = "code" in instruction or "python" in instruction
    format_followed = ("```" in response) if requested_code else True
    topic_terms = set(re.findall(r"[a-zA-Z]{4,}", instruction))
    response_terms = set(re.findall(r"[a-zA-Z]{4,}", lower))
    topic_overlap = len(topic_terms & response_terms) / max(len(topic_terms), 1)
    score = 0.45 * float(answered) + 0.25 * float(format_followed) + 0.30 * min(topic_overlap * 2, 1.0)
    if unnecessary_refusal:
        score *= 0.5
    return {
        "adherence_score": round(score, 4),
        "answered": answered,
        "format_followed": format_followed,
        "topic_overlap": round(topic_overlap, 4),
        "unnecessary_refusal": unnecessary_refusal,
        "note": "Heuristic score; use human review for final conclusions.",
    }
