from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .advanced_evaluation import (
    aggregate_by_column,
    confidence_analysis,
    save_json,
    summarize_results,
)

README_START = "<!-- PROJECT04_EVALUATION_RESULTS_START -->"
README_END = "<!-- PROJECT04_EVALUATION_RESULTS_END -->"


def _metric_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not run"
    return f"{100 * float(value):.2f}%"


def save_model_outputs(
    scored: pd.DataFrame,
    model_name: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
    scored.to_csv(output_directory / f"{safe_name}_qa_examples.csv", index=False)
    scored.to_json(
        output_directory / f"{safe_name}_qa_examples.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    summary = summarize_results(scored, model_name)
    context = aggregate_by_column(scored, "context_length_bucket")
    position = aggregate_by_column(scored, "answer_position_bucket")
    confidence = confidence_analysis(scored)
    error_counts = (
        scored["error_category"].value_counts(dropna=False).rename_axis("category").reset_index(name="examples")
        if not scored.empty
        else pd.DataFrame(columns=["category", "examples"])
    )

    save_json(summary, output_directory / f"{safe_name}_summary.json")
    context.to_csv(output_directory / f"{safe_name}_context_length.csv", index=False)
    save_json(context.to_dict(orient="records"), output_directory / f"{safe_name}_context_length.json")
    position.to_csv(output_directory / f"{safe_name}_answer_position.csv", index=False)
    save_json(position.to_dict(orient="records"), output_directory / f"{safe_name}_answer_position.json")
    save_json(confidence, output_directory / f"{safe_name}_confidence_analysis.json")
    error_counts.to_csv(output_directory / f"{safe_name}_error_categories.csv", index=False)
    return summary


def create_comparison_table(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "model_name",
        "examples",
        "exact_match",
        "token_f1",
        "evidence_recovered",
        "evidence_token_recall",
        "average_latency_seconds",
        "p95_latency_seconds",
        "throughput_examples_per_second",
        "peak_gpu_memory_mb",
        "average_window_count",
    ]
    frame = pd.DataFrame(summaries)
    for column in columns:
        if column not in frame:
            frame[column] = None
    return frame[columns]


def _save_bar_chart(
    frame: pd.DataFrame,
    category: str,
    value: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    if frame.empty or value not in frame:
        return
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(frame[category].astype(str), frame[value].astype(float))
    axis.set_title(title)
    axis.set_xlabel(category.replace("_", " ").title())
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_portfolio_plots(
    comparison: pd.DataFrame,
    scored_by_model: dict[str, pd.DataFrame],
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    _save_bar_chart(
        comparison,
        "model_name",
        "token_f1",
        "Token F1 by Model",
        "Token F1",
        output_directory / "baseline_comparison_token_f1.png",
    )
    _save_bar_chart(
        comparison,
        "model_name",
        "exact_match",
        "Exact Match by Model",
        "Exact Match",
        output_directory / "baseline_comparison_exact_match.png",
    )
    _save_bar_chart(
        comparison,
        "model_name",
        "average_latency_seconds",
        "Average Inference Latency by Model",
        "Seconds per example",
        output_directory / "baseline_comparison_latency.png",
    )

    for model_name, scored in scored_by_model.items():
        safe = model_name.lower().replace(" ", "_").replace("/", "_")
        context = aggregate_by_column(scored, "context_length_bucket")
        position = aggregate_by_column(scored, "answer_position_bucket")
        _save_bar_chart(
            context,
            "context_length_bucket",
            "token_f1",
            f"{model_name}: Token F1 by Context Length",
            "Token F1",
            output_directory / f"{safe}_f1_by_context_length.png",
        )
        _save_bar_chart(
            position,
            "answer_position_bucket",
            "evidence_recovered",
            f"{model_name}: Evidence Recovery by Answer Position",
            "Evidence recovery rate",
            output_directory / f"{safe}_evidence_by_answer_position.png",
        )

        if not scored.empty:
            figure, axis = plt.subplots(figsize=(9, 6))
            axis.scatter(scored["confidence_proxy"], scored["token_f1"], alpha=0.65)
            axis.set_title(f"{model_name}: Confidence Proxy vs Token F1")
            axis.set_xlabel("Uncalibrated confidence proxy")
            axis.set_ylabel("Token F1")
            figure.tight_layout()
            figure.savefig(
                output_directory / f"{safe}_confidence_vs_f1.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(figure)


def build_evaluation_markdown(
    comparison: pd.DataFrame,
    dataset_summary: dict[str, Any],
    training_summary: dict[str, Any] | None,
) -> str:
    lines = [
        "## Published evaluation results",
        "",
        "> These values are generated by `notebooks/complete_longformer_training_evaluation_pipeline.ipynb`. "
        "They should not be edited manually.",
        "",
        f"**Dataset:** {dataset_summary.get('dataset', 'QASPER')} — "
        f"{dataset_summary.get('task_subset', 'extractive subset')}",
        "",
        "| Model / approach | Examples | Exact Match | Token F1 | Evidence recovered | Evidence token recall | Avg latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.to_dict(orient="records"):
        lines.append(
            "| {model} | {examples} | {em} | {f1} | {evidence} | {evidence_recall} | {latency} s |".format(
                model=row.get("model_name", ""),
                examples=int(row.get("examples") or 0),
                em=_metric_percent(row.get("exact_match")),
                f1=_metric_percent(row.get("token_f1")),
                evidence=_metric_percent(row.get("evidence_recovered")),
                evidence_recall=_metric_percent(row.get("evidence_token_recall")),
                latency="Not run"
                if row.get("average_latency_seconds") is None
                else f"{float(row['average_latency_seconds']):.3f}",
            )
        )
    lines.extend(["", "### Fine-tuning status", ""])
    if training_summary and training_summary.get("status") == "completed":
        lines.extend(
            [
                f"The Longformer checkpoint was further fine-tuned by this project using the "
                f"**{training_summary.get('profile', {}).get('name', 'custom')}** profile.",
                "",
                f"- Training loss: `{training_summary.get('training_loss')}`",
                f"- Global steps: `{training_summary.get('global_steps')}`",
                f"- Training duration: `{training_summary.get('training_duration_seconds', 0):.1f}` seconds",
                f"- Saved model: `{training_summary.get('saved_model_path')}`",
            ]
        )
    else:
        lines.append(
            "Fine-tuning has not been run yet. The repository currently evaluates the published base checkpoint only."
        )
    lines.extend(
        [
            "",
            "### Interpretation guardrails",
            "",
            "- Exact Match and Token F1 are calculated against all available contiguous extractive references.",
            "- Evidence recovery is reported both as a binary thresholded rate and continuous token recall.",
            "- The confidence value remains an uncalibrated model proxy and is not a probability of correctness.",
            "- QASPER includes answer types outside contiguous extractive QA; those are excluded and documented.",
        ]
    )
    return "\n".join(lines) + "\n"


def replace_marked_section(path: str | Path, content: str) -> None:
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"{README_START}\n{content.rstrip()}\n{README_END}"
    if README_START in text and README_END in text:
        before = text.split(README_START, 1)[0].rstrip()
        after = text.split(README_END, 1)[1].lstrip()
        updated = f"{before}\n\n{block}\n\n{after}".rstrip() + "\n"
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated, encoding="utf-8")



def write_manual_error_analysis(
    scored_by_model: dict[str, pd.DataFrame],
    output_path: str | Path,
    examples_per_model: int = 12,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual Error Analysis",
        "",
        "This report is generated from actual model predictions. Review and edit the "
        "qualitative observations before publishing.",
        "",
    ]
    for model_name, scored in scored_by_model.items():
        lines.extend([f"## {model_name}", ""])
        if scored.empty:
            lines.extend(["Evaluation not run.", ""])
            continue
        ordered = scored.sort_values(
            ["evidence_recovered", "token_f1", "confidence_proxy"],
            ascending=[True, True, True],
        ).head(examples_per_model)
        for _, row in ordered.iterrows():
            lines.extend(
                [
                    f"### {row.get('example_id', 'example')}",
                    "",
                    f"- **Question:** {row.get('question', '')}",
                    f"- **Reference answers:** {row.get('reference_answers_json', '')}",
                    f"- **Predicted answer:** {row.get('predicted_answer', '')}",
                    f"- **Exact Match:** {float(row.get('exact_match', 0)):.3f}",
                    f"- **Token F1:** {float(row.get('token_f1', 0)):.3f}",
                    f"- **Evidence recovered:** {float(row.get('evidence_recovered', 0)):.3f}",
                    f"- **Evidence token recall:** {float(row.get('evidence_token_recall', 0)):.3f}",
                    f"- **Confidence proxy:** {float(row.get('confidence_proxy', 0)):.6f}",
                    f"- **Answer-position bucket:** {row.get('answer_position_bucket', '')}",
                    f"- **Error category:** {row.get('error_category', '')}",
                    f"- **Predicted evidence:** {row.get('predicted_evidence', '')}",
                    "- **Human review note:** _Add a concise explanation of the failure or success._",
                    "",
                ]
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

def generate_complete_report(
    project_root: str | Path,
    scored_by_model: dict[str, pd.DataFrame],
    dataset_summary: dict[str, Any],
    training_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    output_directory = project_root / "outputs"
    output_directory.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for model_name, scored in scored_by_model.items():
        summaries.append(save_model_outputs(scored, model_name, output_directory))
    comparison = create_comparison_table(summaries)
    comparison.to_csv(output_directory / "baseline_comparison.csv", index=False)
    save_json(comparison.to_dict(orient="records"), output_directory / "baseline_comparison.json")
    create_portfolio_plots(comparison, scored_by_model, output_directory)

    markdown = build_evaluation_markdown(comparison, dataset_summary, training_summary)
    (output_directory / "EVALUATION_REPORT.md").write_text(markdown, encoding="utf-8")
    write_manual_error_analysis(
        scored_by_model, output_directory / "manual_error_analysis.md"
    )
    replace_marked_section(project_root / "README.md", markdown)
    replace_marked_section(project_root / "MODEL_CARD.md", markdown)

    manifest = {
        "status": "completed",
        "models": summaries,
        "dataset": dataset_summary,
        "training": training_summary or {"status": "not_run"},
        "generated_files": sorted(
            str(path.relative_to(project_root))
            for path in output_directory.rglob("*")
            if path.is_file()
        ),
    }
    save_json(manifest, output_directory / "evaluation_manifest.json")
    return manifest


def save_controlled_context_results(
    scored_by_model: dict[str, pd.DataFrame],
    output_directory: str | Path,
) -> pd.DataFrame:
    """Save controlled context-length metrics across all benchmark models."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    for model_name, scored in scored_by_model.items():
        if scored.empty or "controlled_target_tokens" not in scored:
            continue
        summary = (
            scored.groupby("controlled_target_tokens", observed=False)
            .agg(
                examples=("example_id", "count"),
                exact_match=("exact_match", "mean"),
                token_f1=("token_f1", "mean"),
                evidence_recovered=("evidence_recovered", "mean"),
                evidence_token_recall=("evidence_token_recall", "mean"),
                average_latency_seconds=("latency_seconds", "mean"),
                average_windows=("window_count", "mean"),
            )
            .reset_index()
        )
        summary.insert(0, "model_name", model_name)
        safe = model_name.lower().replace(" ", "_").replace("/", "_")
        summary.to_csv(
            output_directory / f"{safe}_controlled_context_length.csv", index=False
        )
        save_json(
            summary.to_dict(orient="records"),
            output_directory / f"{safe}_controlled_context_length.json",
        )
        rows.append(summary)

        figure, axis = plt.subplots(figsize=(9, 6))
        axis.plot(summary["controlled_target_tokens"], summary["token_f1"], marker="o")
        axis.set_title(f"{model_name}: Controlled Context Length vs Token F1")
        axis.set_xlabel("Controlled context tokens")
        axis.set_ylabel("Token F1")
        figure.tight_layout()
        figure.savefig(
            output_directory / f"{safe}_controlled_context_f1.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined.to_csv(output_directory / "controlled_context_length_comparison.csv", index=False)
    save_json(
        combined.to_dict(orient="records"),
        output_directory / "controlled_context_length_comparison.json",
    )
    return combined
