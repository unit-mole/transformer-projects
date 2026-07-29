from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_metric_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = ["recall_at_10", "mrr_at_10", "ndcg_at_10", "map_at_100"]
    existing = [metric for metric in metrics if metric in summary.columns]
    frame = summary.pivot(index="approach", columns="dataset", values=existing)
    ax = frame.plot(kind="bar", figsize=(13, 7), rot=20)
    ax.set_title("Retrieval and Reranking Quality by Dataset")
    ax.set_ylabel("Metric score")
    ax.set_ylim(bottom=0)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def plot_recall_curves(summary: pd.DataFrame, output_path: Path) -> None:
    recall_columns = sorted(
        [column for column in summary.columns if column.startswith("recall_at_")],
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    figure, axis = plt.subplots(figsize=(11, 7))
    for (dataset, approach), row in summary.set_index(["dataset", "approach"]).iterrows():
        x_values = [int(column.rsplit("_", 1)[1]) for column in recall_columns]
        y_values = [row[column] for column in recall_columns]
        axis.plot(x_values, y_values, marker="o", label=f"{dataset} · {approach}")
    axis.set_title("Recall@K Candidate Coverage")
    axis.set_xlabel("K")
    axis.set_ylabel("Recall")
    axis.set_ylim(0, 1.02)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_latency(latency: pd.DataFrame, output_path: Path) -> None:
    frame = latency.copy()
    frame["label"] = frame["dataset"] + " · " + frame["approach"]
    ax = frame.plot(
        x="label",
        y="mean_query_ms",
        kind="bar",
        figsize=(12, 6),
        rot=25,
        legend=False,
    )
    ax.set_title("Average Query Latency")
    ax.set_xlabel("")
    ax.set_ylabel("Milliseconds")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def plot_reranking_delta(per_query: pd.DataFrame, output_path: Path) -> None:
    if "ndcg_delta" not in per_query.columns:
        return
    ax = per_query["ndcg_delta"].plot(
        kind="hist",
        bins=25,
        figsize=(10, 6),
        title="Per-Query nDCG@10 Change After Reranking",
    )
    ax.set_xlabel("Reranked nDCG@10 − Bi-encoder nDCG@10")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def make_portfolio_markdown(
    summary: pd.DataFrame,
    bootstrap: dict[str, Any],
    metadata: list[dict[str, Any]],
) -> str:
    key_columns = [
        "dataset",
        "approach",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "map_at_100",
        "mean_query_ms",
    ]
    present = [column for column in key_columns if column in summary.columns]
    table = summary[present].copy()
    for column in table.columns:
        if column not in {"dataset", "approach"}:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(4)

    lines = [
        "# Project 03 Benchmark Results",
        "",
        "> Generated from actual model execution. Do not edit metric values manually.",
        "",
        "## Datasets",
        "",
    ]
    for item in metadata:
        lines.append(
            f"- **{item['dataset']} ({item['split']}):** "
            f"{item['corpus_documents']:,} documents, "
            f"{item['queries']:,} evaluated queries and "
            f"{item['relevance_judgments']:,} relevance judgments."
        )
    lines.extend(["", "## Model and baseline comparison", "", table.to_markdown(index=False)])

    lines.extend(["", "## Paired bootstrap reranking analysis", ""])
    for dataset, metrics in bootstrap.items():
        lines.append(f"### {dataset}")
        lines.append("")
        for metric, result in metrics.items():
            lines.append(
                f"- **{metric}:** mean delta `{result['mean_delta']:.4f}`, "
                f"95% CI `[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]`, "
                f"P(delta > 0) `{result['probability_delta_positive']:.3f}`."
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation guidance",
            "",
            "- Recall@K measures first-stage candidate coverage.",
            "- MRR@10 rewards placing the first relevant result early.",
            "- nDCG@10 evaluates the complete top-ranked ordering.",
            "- MAP@100 summarizes precision across all relevant results.",
            "- A positive reranking delta means the cross-encoder improved the ranking on average.",
            "- Latency values depend on the local GPU, CPU, drivers and batch sizes.",
            "",
        ]
    )
    return "\n".join(lines)


def update_latest_alias(run_dir: Path, benchmark_root: Path) -> Path:
    latest = benchmark_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)
    return latest
