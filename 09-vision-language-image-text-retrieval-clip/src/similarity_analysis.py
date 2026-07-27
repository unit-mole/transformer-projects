from __future__ import annotations

from typing import Any
import numpy as np


def summarize_similarity(scores: list[float]) -> dict[str, Any]:
    if not scores:
        raise ValueError("scores cannot be empty")
    array = np.asarray(scores, dtype=float)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std()),
        "top_margin": float(np.sort(array)[-1] - np.sort(array)[-2]) if array.size > 1 else None,
    }
