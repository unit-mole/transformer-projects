from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.runner import BenchmarkConfig, run_benchmark_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run large-scale lexical, dense and cross-encoder benchmarks."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "nfcorpus"],
        help="Official BEIR dataset names.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--bi-encoder-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--cross-encoder-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    parser.add_argument("--model-label", default="base_minilm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rerank-k", type=int, default=100)
    parser.add_argument("--bi-batch-size", type=int, default=128)
    parser.add_argument("--cross-batch-size", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-tfidf", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--skip-md5", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        project_root=PROJECT_ROOT,
        datasets=tuple(args.datasets),
        split=args.split,
        bi_encoder_model=args.bi_encoder_model,
        cross_encoder_model=args.cross_encoder_model,
        model_label=args.model_label,
        device=args.device,
        candidate_k=args.candidate_k,
        rerank_k=args.rerank_k,
        bi_encoder_batch_size=args.bi_batch_size,
        cross_encoder_batch_size=args.cross_batch_size,
        max_queries=args.max_queries,
        bootstrap_samples=args.bootstrap_samples,
        verify_md5=not args.skip_md5,
        run_tfidf=not args.skip_tfidf,
        run_bm25=not args.skip_bm25,
        run_cross_encoder=not args.skip_reranker,
        run_name=args.run_name,
    )
    result = run_benchmark_suite(config)
    print("\nBenchmark completed.")
    print(f"Run directory: {result.run_directory}")
    print(f"Latest results: {result.latest_directory}")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
