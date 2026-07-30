"""Evaluation pipeline for instruction adherence, relevance, hallucination risk, BERTScore, and latency."""
from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .bertscore_evaluation import calculate_bertscore
from .data_preprocessing import load_jsonl
from .hallucination_analysis import analyze_hallucination_risk
from .inference_pipeline import InstructionAssistant
from .instruction_adherence import evaluate_instruction_adherence
from .relevance_scoring import score_relevance


def evaluate_records(
    records: Iterable[Dict[str, object]],
    assistant: InstructionAssistant,
    *,
    limit: Optional[int] = None,
    include_bertscore: bool = False,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    selected = list(records)[:limit] if limit else list(records)

    for record in selected:
        instruction = str(record.get("instruction", ""))
        input_text = str(record.get("input", ""))
        category = str(record.get("category", "Concept explanation"))
        reference = str(record.get("output", record.get("reference_answer", "")))
        started = time.perf_counter()
        result = assistant.generate(instruction, category, input_text)
        measured = time.perf_counter() - started
        response = str(result["response"])
        adherence = evaluate_instruction_adherence(instruction, response)
        relevance = score_relevance(instruction + " " + input_text, response)
        hallucination = analyze_hallucination_risk(instruction, response, reference)
        rows.append({
            "id": record.get("id", ""),
            "prompt": instruction,
            "input": input_text,
            "category": category,
            "reference_answer": reference,
            "generated_answer": response,
            "adherence_score": adherence["adherence_score"],
            "relevance_score": relevance["combined_relevance"],
            "hallucination_flag": hallucination["hallucination_flag"],
            "hallucination_severity": hallucination["severity"],
            "latency_seconds": round(float(result.get("latency_seconds", measured)), 4),
            "model_mode": result.get("model_mode", "unknown"),
            "reviewer_notes": "",
        })

    bertscore_result: Dict[str, object] = {"status": "not_requested"}
    if include_bertscore and rows and all(row["reference_answer"] for row in rows):
        bertscore_result = calculate_bertscore(
            [str(row["generated_answer"]) for row in rows],
            [str(row["reference_answer"]) for row in rows],
        )
        for row, p, r, f in zip(rows, bertscore_result["precision"], bertscore_result["recall"], bertscore_result["f1"]):
            row["bertscore_precision"] = p
            row["bertscore_recall"] = r
            row["bertscore_f1"] = f

    summary = {
        "status": "completed" if rows else "no_records",
        "evaluated_examples": len(rows),
        "average_instruction_adherence": round(statistics.mean(float(r["adherence_score"]) for r in rows), 4) if rows else None,
        "average_response_relevance": round(statistics.mean(float(r["relevance_score"]) for r in rows), 4) if rows else None,
        "hallucination_flag_rate": round(sum(bool(r["hallucination_flag"]) for r in rows) / len(rows), 4) if rows else None,
        "average_latency_seconds": round(statistics.mean(float(r["latency_seconds"]) for r in rows), 4) if rows else None,
        "bertscore": bertscore_result,
        "warning": "Heuristic adherence, relevance, and hallucination outputs require human review.",
    }
    return rows, summary


def save_evaluation(rows: List[Dict[str, object]], summary: Dict[str, object], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output / "manual_review_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (output / "model_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "instruction_adherence_results.json").write_text(
        json.dumps({"status": summary["status"], "average": summary["average_instruction_adherence"]}, indent=2), encoding="utf-8"
    )
    (output / "response_relevance_results.json").write_text(
        json.dumps({"status": summary["status"], "average": summary["average_response_relevance"], "method": "TF-IDF and lexical heuristic"}, indent=2), encoding="utf-8"
    )
    (output / "bertscore_results.json").write_text(json.dumps(summary["bertscore"], indent=2), encoding="utf-8")


def evaluate_from_file(
    evaluation_path: str | Path,
    output_dir: str | Path,
    *,
    limit: Optional[int] = None,
    include_bertscore: bool = False,
) -> Dict[str, object]:
    records = load_jsonl(evaluation_path)
    assistant = InstructionAssistant()
    rows, summary = evaluate_records(records, assistant, limit=limit, include_bertscore=include_bertscore)
    save_evaluation(rows, summary, output_dir)
    return summary
