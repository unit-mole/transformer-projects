from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_metrics import evaluate_ranking, summarize_query_metrics
from src.retrievers import CrossEncoderReranker, DenseRetriever, HashRetriever, TfidfRetriever


def chunk_text(chunk: dict) -> str:
    keywords = " ".join(chunk.get("keywords", []))
    return f"{chunk['projectName']}\n{chunk['section']}\n{chunk['text']}\n{keywords}".strip()


def benchmark_method(
    name: str,
    search: Callable[[str, int], object],
    chunks: list[dict],
    questions: list[dict],
    k_values: list[int],
    max_candidates: int,
) -> dict:
    details: list[dict] = []
    summaries: dict[str, dict] = {}
    embedding_latencies: list[float] = []
    retrieval_latencies: list[float] = []

    answerable_questions = [item for item in questions if item.get("answerable", True) and item.get("expected_source_project_ids")]
    for k in k_values:
        rows = []
        for item in answerable_questions:
            result = search(item["question"], max(max_candidates, k))
            retrieved_projects = [chunks[index]["projectId"] for index in result.indices[:k]]
            row = evaluate_ranking(
                question_id=item["id"],
                relevant_ids=item.get("expected_source_project_ids", []),
                retrieved_ids=retrieved_projects,
                k=k,
            )
            rows.append(row)
            details.append({"method": name, **row.to_dict()})
            embedding_latencies.append(float(result.query_latency_ms))
            retrieval_latencies.append(float(result.retrieval_latency_ms))
        summaries[f"k={k}"] = summarize_query_metrics(rows)

    return {
        "method": name,
        "summary": summaries,
        "latency_ms": {
            "query_embedding_mean": round(statistics.fmean(embedding_latencies), 3) if embedding_latencies else 0.0,
            "query_embedding_p95": round(float(np.percentile(embedding_latencies, 95)), 3) if embedding_latencies else 0.0,
            "retrieval_mean": round(statistics.fmean(retrieval_latencies), 3) if retrieval_latencies else 0.0,
            "retrieval_p95": round(float(np.percentile(retrieval_latencies, 95)), 3) if retrieval_latencies else 0.0,
        },
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare keyword, hash, MiniLM, E5, and reranked retrieval.")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/processed/document_chunks.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/processed/evaluation_questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/retrieval_benchmark.json")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-transformers", action="store_true")
    parser.add_argument("--include-e5", action="store_true")
    parser.add_argument("--include-reranker", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=20)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    documents = [chunk_text(chunk) for chunk in chunks]

    methods: list[dict] = []

    tfidf = TfidfRetriever(documents)
    methods.append(benchmark_method("tfidf-keyword", tfidf.search, chunks, questions, args.k, args.candidate_k))

    hash_retriever = HashRetriever(documents)
    methods.append(benchmark_method("hash-vector", hash_retriever.search, chunks, questions, args.k, args.candidate_k))

    if not args.skip_transformers:
        minilm = DenseRetriever(
            documents,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device=args.device,
        )
        methods.append(benchmark_method("minilm-dense", minilm.search, chunks, questions, args.k, args.candidate_k))

        if args.include_e5:
            e5 = DenseRetriever(
                documents,
                model_name="intfloat/e5-small-v2",
                device=args.device,
                query_prefix="query: ",
                passage_prefix="passage: ",
            )
            methods.append(benchmark_method("e5-small-v2", e5.search, chunks, questions, args.k, args.candidate_k))

        if args.include_reranker:
            reranker = CrossEncoderReranker(
                "cross-encoder/ms-marco-MiniLM-L6-v2",
                device=args.device,
            )

            def reranked_search(question: str, top_k: int):
                dense_result = minilm.search(question, args.candidate_k)
                rerank_result = reranker.rerank(
                    question,
                    dense_result.indices,
                    documents,
                    top_k=top_k,
                )
                from src.retrievers import RankedResult
                return RankedResult(
                    indices=rerank_result.indices,
                    scores=rerank_result.scores,
                    query_latency_ms=dense_result.query_latency_ms,
                    retrieval_latency_ms=dense_result.retrieval_latency_ms + rerank_result.retrieval_latency_ms,
                )

            methods.append(benchmark_method(
                "minilm-plus-cross-encoder",
                reranked_search,
                chunks,
                questions,
                args.k,
                args.candidate_k,
            ))

    payload = {
        "status": "measured",
        "question_count": len([item for item in questions if item.get("answerable", True)]),
        "unsupported_question_count": len([item for item in questions if not item.get("answerable", True)]),
        "chunk_count": len(chunks),
        "k_values": args.k,
        "methods": methods,
        "metric_definitions": {
            "hit_rate": "Fraction of questions with at least one relevant project in top K.",
            "precision": "Mean fraction of the K retrieved projects that are relevant.",
            "recall": "Mean fraction of all expected relevant projects retrieved in top K.",
            "mrr": "Mean reciprocal rank of the first relevant project.",
            "map": "Mean average precision at K.",
            "ndcg": "Mean normalized discounted cumulative gain at K.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved retrieval benchmark to {args.output}")
    for method in methods:
        print(method["method"], method["summary"].get("k=5"))


if __name__ == "__main__":
    main()
