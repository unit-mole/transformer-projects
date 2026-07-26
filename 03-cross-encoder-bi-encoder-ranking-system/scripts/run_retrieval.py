from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ranking_engine import TwoStageRankingEngine
from src.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bi-encoder retrieval.")
    parser.add_argument("query", nargs="?", default="How can I find similar quality complaints?")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    engine = TwoStageRankingEngine.from_settings(Settings.from_yaml())
    response = engine.retrieve(args.query, candidate_k=args.top_k)
    columns = ["retrieval_rank", "document_id", "title", "bi_encoder_score"]
    print(response.candidates[columns].to_string(index=False))
    print(f"\nRetrieval latency: {response.latency.retrieval_ms:.2f} ms")


if __name__ == "__main__":
    main()
