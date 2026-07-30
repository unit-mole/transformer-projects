"""Heuristic instruction-adherence rubric for educational responses."""
from __future__ import annotations

import re
from typing import Dict

from .relevance_scoring import score_relevance

ML_TERMS = {
    "model", "data", "machine", "learning", "algorithm", "metric", "training", "validation",
    "feature", "classification", "regression", "neural", "transformer", "precision", "recall",
    "deployment", "pipeline", "dataset", "prediction", "cluster", "loss", "accuracy", "lora", "peft",
}


def _format_score(instruction: str, response: str) -> float:
    lower = instruction.lower()
    if "compare" in lower:
        comparison_markers = ("while", "whereas", "difference", "compared", "both")
        return 1.0 if any(marker in response.lower() for marker in comparison_markers) else 0.5
    if "code" in lower or "python" in lower:
        return 1.0 if "```" in response or "import " in response else 0.4
    if "one sentence" in lower:
        sentence_count = len([s for s in re.split(r"[.!?]+", response) if s.strip()])
        return 1.0 if sentence_count <= 1 else 0.4
    return 1.0


def evaluate_instruction_adherence(instruction: str, response: str) -> Dict[str, float | bool | str]:
    nonempty = float(bool(response.strip()))
    relevance = float(score_relevance(instruction, response)["combined_relevance"])
    topic_tokens = set(re.findall(r"[a-z]+", (instruction + " " + response).lower()))
    in_scope = bool(topic_tokens & ML_TERMS)
    format_following = _format_score(instruction, response)
    refusal_of_safe_prompt = bool(re.search(r"\b(i cannot|i can't|unable to help)\b", response.lower()))
    score = max(0.0, min(1.0, 0.25 * nonempty + 0.35 * relevance + 0.25 * format_following + 0.15 * float(in_scope) - 0.25 * float(refusal_of_safe_prompt)))
    return {
        "adherence_score": round(score, 4),
        "answers_request": relevance >= 0.12 and nonempty == 1.0,
        "follows_requested_format": format_following >= 0.8,
        "stays_in_ml_ds_scope": in_scope,
        "safe_prompt_refusal": refusal_of_safe_prompt,
        "method": "documented_heuristic_requires_manual_review",
    }
