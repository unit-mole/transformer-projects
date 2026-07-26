from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceProxy:
    score: float
    method: str
    label: str
    explanation: str


def _label(score: float) -> str:
    if score >= 0.75:
        return "higher proxy"
    if score >= 0.45:
        return "medium proxy"
    return "lower proxy"


def sequence_score_confidence(raw_sequence_score: float) -> ConfidenceProxy:
    """Convert a normalized log-sequence score into a bounded proxy."""
    raw = float(raw_sequence_score)
    if raw <= 0:
        score = math.exp(max(raw, -20.0))
    else:
        score = 1.0 / (1.0 + math.exp(-raw))
    score = max(0.0, min(1.0, score))
    return ConfidenceProxy(
        score=score,
        method="exponentiated_normalized_sequence_score",
        label=_label(score),
        explanation=(
            "Derived from the model's normalized generation sequence score. "
            "It is not calibrated to translation correctness."
        ),
    )


def heuristic_confidence(
    source_text: str,
    translated_text: str,
    *,
    unknown_token_count: int = 0,
) -> ConfidenceProxy:
    source_tokens = source_text.split()
    target_tokens = translated_text.split()

    if not source_tokens or not target_tokens:
        score = 0.0
    else:
        ratio = len(target_tokens) / max(len(source_tokens), 1)
        length_component = max(0.0, 1.0 - min(abs(math.log(max(ratio, 1e-6))), 1.0))

        normalized = [re.sub(r"\W+", "", token.lower()) for token in target_tokens]
        normalized = [token for token in normalized if token]
        unique_ratio = len(set(normalized)) / max(len(normalized), 1)
        repetition_component = min(1.0, unique_ratio + 0.20)

        unknown_penalty = min(0.5, unknown_token_count * 0.1)
        score = (0.55 * length_component) + (0.45 * repetition_component)
        score = max(0.0, min(1.0, score - unknown_penalty))

    return ConfidenceProxy(
        score=score,
        method="length_repetition_unknown_token_heuristic",
        label=_label(score),
        explanation=(
            "Fallback heuristic using output length, repetition, and unknown tokens. "
            "It is not a probability of correctness."
        ),
    )
