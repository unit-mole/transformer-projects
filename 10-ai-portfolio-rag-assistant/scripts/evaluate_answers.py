from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.answer_evaluation import NliGroundednessEvaluator, summarize_answer_evaluations
from src.local_generator import LocalInstructionGenerator, grounded_extractive_answer
from src.retrievers import DenseRetriever


def chunk_text(chunk: dict) -> str:
    return f"{chunk['projectName']}\n{chunk['section']}\n{chunk['text']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers and evaluate groundedness/citation correctness.")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/processed/document_chunks.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/processed/evaluation_questions.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--generator", choices=["extractive", "flan-t5-base"], default="extractive")
    parser.add_argument("--retriever-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--min-retrieval-score", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    documents = [chunk_text(chunk) for chunk in chunks]
    retriever = DenseRetriever(documents, args.retriever_model, device=args.device)
    nli = NliGroundednessEvaluator(args.nli_model, device=args.device)

    generator = None
    if args.generator == "flan-t5-base":
        generator = LocalInstructionGenerator("google/flan-t5-base", device=args.device)

    answer_rows = []
    evaluation_rows = []
    latencies = []

    for item in questions:
        result = retriever.search(item["question"], args.top_k)
        retrieved = [
            {**chunks[index], "_retrievalScore": float(score)}
            for index, score in zip(result.indices, result.scores)
        ]
        if generator is None:
            generated = grounded_extractive_answer(
                item["question"], retrieved, min_retrieval_score=args.min_retrieval_score
            )
        else:
            generated = generator.generate(item["question"], retrieved)

        citation_evidence = {f"S{index}": chunk["text"] for index, chunk in enumerate(retrieved, start=1)}
        evaluation = nli.evaluate(
            question_id=item["id"],
            question=item["question"],
            answer=generated.text,
            citation_evidence=citation_evidence,
            all_retrieved_evidence=[chunk["text"] for chunk in retrieved],
            answerable=item.get("answerable"),
        )
        evaluation_rows.append(evaluation)
        total_ms = result.query_latency_ms + result.retrieval_latency_ms + generated.latency_ms
        latencies.append({
            "question_id": item["id"],
            "query_embedding_ms": round(result.query_latency_ms, 3),
            "retrieval_ms": round(result.retrieval_latency_ms, 3),
            "generation_ms": round(generated.latency_ms, 3),
            "total_ms": round(total_ms, 3),
            "top_k": args.top_k,
            "corpus_size": len(chunks),
            "generator_mode": generated.mode,
        })
        answer_rows.append({
            "question_id": item["id"],
            "question": item["question"],
            "answer": generated.text,
            "retrieved_project_ids": [chunk["projectId"] for chunk in retrieved],
            "groundedness_score": evaluation.groundedness_score,
            "citation_precision": evaluation.citation_precision,
            "citation_completeness": evaluation.citation_completeness,
            "unsupported_claim_rate": evaluation.unsupported_claim_rate,
            "generator_mode": generated.mode,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_answer_evaluations(evaluation_rows)

    groundedness_payload = {
        "status": "measured",
        "metric": "claim_level_nli_groundedness",
        "nli_model": args.nli_model,
        "entailment_threshold": nli.entailment_threshold,
        "summary": summary,
        "records": [row.to_dict() for row in evaluation_rows],
    }
    (args.output_dir / "answer_groundedness_results.json").write_text(
        json.dumps(groundedness_payload, indent=2), encoding="utf-8"
    )

    citation_payload = {
        "status": "measured",
        "metric": "claim_level_citation_correctness",
        "summary": {
            "mean_citation_precision": summary["mean_citation_precision"],
            "mean_citation_completeness": summary["mean_citation_completeness"],
            "mean_unsupported_claim_rate": summary["mean_unsupported_claim_rate"],
        },
        "records": [row.to_dict() for row in evaluation_rows],
    }
    (args.output_dir / "citation_correctness_results.json").write_text(
        json.dumps(citation_payload, indent=2), encoding="utf-8"
    )

    latency_values = [row["total_ms"] for row in latencies]
    latency_payload = {
        "status": "measured_locally",
        "summary": {
            "count": len(latency_values),
            "mean_ms": round(float(np.mean(latency_values)), 3) if latency_values else 0.0,
            "median_ms": round(float(np.median(latency_values)), 3) if latency_values else 0.0,
            "p90_ms": round(float(np.percentile(latency_values, 90)), 3) if latency_values else 0.0,
            "p95_ms": round(float(np.percentile(latency_values, 95)), 3) if latency_values else 0.0,
            "min_ms": round(float(np.min(latency_values)), 3) if latency_values else 0.0,
            "max_ms": round(float(np.max(latency_values)), 3) if latency_values else 0.0,
        },
        "records": latencies,
        "note": "Rerun scripts/benchmark_deployed_api.py after Vercel deployment for production latency.",
    }
    (args.output_dir / "response_latency_results.json").write_text(
        json.dumps(latency_payload, indent=2), encoding="utf-8"
    )

    csv_path = args.output_dir / "rag_answer_examples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=answer_rows[0].keys() if answer_rows else ["question_id"])
        writer.writeheader()
        writer.writerows(answer_rows)

    print(json.dumps(summary, indent=2))
    print(f"Saved answer evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
