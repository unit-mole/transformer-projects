from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_model_comparison(frame: pd.DataFrame, output_path: str | Path) -> Path:
    required = {"model", "rouge1", "rouge2", "rougeL"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Comparison frame must contain: {sorted(required)}")
    chart = frame.set_index("model")[["rouge1", "rouge2", "rougeL"]]
    axis = chart.plot(kind="bar", figsize=(10, 6))
    axis.set_ylabel("Score")
    axis.set_title("Summarization Model Comparison")
    axis.set_ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=180)
    plt.close()
    return destination


def plot_latency_by_beam(frame: pd.DataFrame, output_path: str | Path) -> Path:
    required = {"num_beams", "inference_seconds"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Latency frame must contain: {sorted(required)}")
    grouped = frame.groupby("num_beams", as_index=False)["inference_seconds"].mean()
    axis = grouped.plot(x="num_beams", y="inference_seconds", kind="bar", legend=False)
    axis.set_ylabel("Average inference time (seconds)")
    axis.set_title("Inference Time by Beam Count")
    plt.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=180)
    plt.close()
    return destination
