"""Optional visualizations for corpus and retrieval analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def save_histogram(values: Iterable[float], title: str, xlabel: str, output_path: Path) -> None:
    values = list(values)
    if not values:
        raise ValueError("Cannot plot an empty value list")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=min(20, max(5, len(values))))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
