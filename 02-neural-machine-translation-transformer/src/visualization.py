from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_latency_by_length(
    evaluation: pd.DataFrame,
    output_path: str | Path,
) -> str:
    frame = evaluation.copy()
    frame["source_words"] = frame["source_text"].astype(str).str.split().str.len()

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(frame["source_words"], frame["wall_latency_seconds"], alpha=0.7)
    axis.set_xlabel("Source sentence length (words)")
    axis.set_ylabel("Wall latency (seconds)")
    axis.set_title("Translation latency by sentence length")
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return str(destination)


def plot_direction_metrics(
    metrics: dict,
    output_path: str | Path,
) -> str:
    frame = pd.DataFrame(
        [
            {
                "direction": direction,
                "SacreBLEU": values["sacrebleu"],
                "chrF": values["chrf"],
            }
            for direction, values in metrics.items()
        ]
    ).set_index("direction")

    axis = frame.plot(kind="bar", figsize=(8, 5))
    axis.set_ylabel("Score")
    axis.set_title("Direction-wise translation metrics")
    axis.figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    axis.figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(axis.figure)
    return str(destination)
