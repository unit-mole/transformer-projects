from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def save_distribution(values: list[float], title: str, xlabel: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=min(20, max(5, len(values))))
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
