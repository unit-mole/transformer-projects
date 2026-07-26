"""Evaluation visualization helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(matrix, class_names: list[str], output_path: str | Path, title: str = "Confusion Matrix") -> Path:
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(values)
    fig.colorbar(image, ax=ax)
    ax.set(title=title, xlabel="Predicted label", ylabel="True label")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, str(values[row, col]), ha="center", va="center")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
