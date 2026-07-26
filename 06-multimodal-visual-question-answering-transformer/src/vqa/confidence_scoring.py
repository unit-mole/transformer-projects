from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

@dataclass(frozen=True)
class ConfidenceProxy:
    top_probability: float
    runner_up_probability: float
    margin: float
    label: str

def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("values must not be empty")
    maximum = max(values)
    exps = [math.exp(float(v) - maximum) for v in values]
    total = sum(exps)
    return [value / total for value in exps]

def confidence_from_logits(logits: Sequence[float]) -> ConfidenceProxy:
    probabilities = softmax(logits)
    ranked = sorted(probabilities, reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else 0.0
    margin = top - second
    label = "high" if top >= 0.75 and margin >= 0.25 else "medium" if top >= 0.45 else "low"
    return ConfidenceProxy(top, second, margin, label)
