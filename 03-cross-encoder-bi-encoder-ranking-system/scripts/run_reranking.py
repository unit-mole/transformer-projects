from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ranking_engine import TwoStageRankingEngine
from src.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-stage retrieval and reranking.")
    parser.add_argument("query", nargs="?", default="How can I find similar quality complaints?")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--rerank-k", type=int, default=5)
    args = parser.parse_args()

    engine = TwoStageRankingEngine.from_settings(Settings.from_yaml())
    response = engine.search(
        args.query,
        candidate_k=args.candidate_k,
        rerank_k=args.rerank_k,
    )
    columns = [
        "retrieval_rank",
        "reranked_rank",
        "rank_movement",
        "document_id",
        "title",
        "bi_encoder_score",
        "cross_encoder_score",
    ]
    print(response.reranked_results[columns].to_string(index=False))
    print(
        f"\nRetrieval: {response.latency.retrieval_ms:.2f} ms | "
        f"Reranking: {response.latency.reranking_ms:.2f} ms | "
        f"Total: {response.latency.total_search_ms:.2f} ms"
    )


if __name__ == "__main__":
    main()
