"""End-to-end generation and evaluation over the held-out prompt set."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .bertscore_evaluation import calculate_bertscore
from .data_preprocessing import load_jsonl
from .hallucination_analysis import flag_hallucination_risks
from .instruction_adherence import score_instruction_adherence
from .prompt_templates import format_prompt
from .relevance_scoring import score_relevance
from .response_generation import GenerationSettings, generate_response


def evaluate_model(loaded_model, evaluation_path: str | Path, output_dir: str | Path, limit: int | None = None) -> dict[str, Any]:
    rows = load_jsonl(evaluation_path)
    if limit:
        rows = rows[:limit]
    results = []
    for row in rows:
        start = time.perf_counter()
        generated, metadata = generate_response(
            loaded_model,
            format_prompt(row["instruction"], row.get("input", "")),
            GenerationSettings(temperature=0.0),
        )
        latency = time.perf_counter() - start
        adherence = score_instruction_adherence(row["instruction"], generated)
        relevance = score_relevance(row["instruction"], generated, row.get("reference_answer", ""))
        hallucination = flag_hallucination_risks(row["instruction"], generated)
        results.append({
            **row,
            "generated_answer": generated,
            "latency_seconds": round(latency, 4),
            **adherence,
            **relevance,
            **hallucination,
            "model_mode": metadata["model_mode"],
            "reviewer_notes": "",
        })

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output / "generated_response_examples.csv", index=False)
    frame.to_csv(output / "manual_review_results.csv", index=False)

    bert = calculate_bertscore(frame["generated_answer"].tolist(), frame["reference_answer"].tolist()) if len(frame) else {}
    summary = {
        "status": "completed",
        "evaluated_examples": int(len(frame)),
        "model_mode": loaded_model.mode,
        "mean_instruction_adherence": float(frame["adherence_score"].mean()) if len(frame) else None,
        "mean_response_relevance": float(frame["relevance_score"].mean()) if len(frame) else None,
        "hallucination_flag_rate": float(frame["hallucination_flag"].mean()) if len(frame) else None,
        "average_latency_seconds": float(frame["latency_seconds"].mean()) if len(frame) else None,
        "bertscore_average_f1": bert.get("average_f1"),
        "warning": "Heuristic and semantic metrics require human interpretation; they are not factuality guarantees.",
    }
    (output / "model_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "bertscore_results.json").write_text(json.dumps(bert, indent=2), encoding="utf-8")
    return summary
