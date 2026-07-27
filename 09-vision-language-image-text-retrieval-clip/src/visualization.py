from __future__ import annotations

from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt


def plot_similarity_scores(labels: Sequence[str], scores: Sequence[float], output_path: str | Path) -> Path:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must match")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(list(labels), list(scores))
    ax.set_xlabel("Cosine similarity")
    ax.set_title("CLIP similarity-score analysis")
    fig.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target
