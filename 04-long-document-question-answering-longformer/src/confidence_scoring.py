from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def stable_softmax(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return np.asarray([], dtype=np.float64)
    shifted = array - np.max(array)
    exp_values = np.exp(shifted)
    denominator = float(exp_values.sum())
    if denominator == 0.0 or not np.isfinite(denominator):
        return np.zeros_like(array)
    return exp_values / denominator


def span_confidence_proxy(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    valid_positions: list[int],
    start_index: int,
    end_index: int,
) -> float:
    """Return an uncalibrated proxy from start/end token probabilities."""
    if not valid_positions:
        return 0.0
    start_values = start_logits[valid_positions]
    end_values = end_logits[valid_positions]
    start_probs = stable_softmax(start_values)
    end_probs = stable_softmax(end_values)
    position_to_local = {position: index for index, position in enumerate(valid_positions)}
    if start_index not in position_to_local or end_index not in position_to_local:
        return 0.0
    p_start = float(start_probs[position_to_local[start_index]])
    p_end = float(end_probs[position_to_local[end_index]])
    return float(np.clip(math.sqrt(max(p_start, 0.0) * max(p_end, 0.0)), 0.0, 1.0))


def confidence_label(value: float) -> str:
    if value >= 0.20:
        return "higher model confidence proxy"
    if value >= 0.05:
        return "moderate model confidence proxy"
    if value >= 0.01:
        return "low model confidence proxy"
    return "very low model confidence proxy"
