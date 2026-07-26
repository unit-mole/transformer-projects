from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_comparison(summary: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    frame = pd.DataFrame(
        {
            "Approach": ["Bi-Encoder", "Bi-Encoder + Cross-Encoder"],
            "MRR@10": [
                summary["bi_encoder_mrr_at_10"],
                summary["reranked_mrr_at_10"],
            ],
            "nDCG@10": [
                summary["bi_encoder_ndcg_at_10"],
                summary["reranked_ndcg_at_10"],
            ],
        }
    ).set_index("Approach")
    ax = frame.plot(kind="bar", rot=0, title="Ranking Quality Before and After Reranking")
    ax.set_ylabel("Score")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)
    return output_path


def plot_latency(frame: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ax = frame.plot(
        x="candidate_k",
        y=["retrieval_mean_ms", "reranking_mean_ms", "total_mean_ms"],
        marker="o",
        title="Latency by Candidate Set Size",
    )
    ax.set_xlabel("Candidate K")
    ax.set_ylabel("Milliseconds")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)
    return output_path
