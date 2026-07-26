from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_context_length_analysis(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5))
    if summary.empty:
        axis.text(0.5, 0.5, "Run evaluation to generate context-length metrics.", ha="center")
        axis.set_axis_off()
    else:
        axis.plot(summary["context_length_bucket"], summary["token_f1"], marker="o", label="Token F1")
        axis.plot(summary["context_length_bucket"], summary["evidence_recall"], marker="o", label="Evidence Recall")
        axis.set_xlabel("Context length bucket")
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1)
        axis.set_title("Long-document QA performance by context length")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path
