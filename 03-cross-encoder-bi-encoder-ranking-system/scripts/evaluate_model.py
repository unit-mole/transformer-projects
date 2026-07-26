from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_evaluation import evaluate_engine
from src.ranking_engine import TwoStageRankingEngine
from src.settings import Settings
from src.visualization import plot_metric_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and reranking.")
    parser.add_argument("--split", default=None, choices=["test", "validation"])
    parser.add_argument("--candidate-k", type=int, default=10)
    args = parser.parse_args()

    settings = Settings.from_yaml()
    engine = TwoStageRankingEngine.from_settings(settings)
    details, summary = evaluate_engine(
        engine,
        split=args.split,
        candidate_k=args.candidate_k,
    )

    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    details.to_csv(outputs / "ranking_examples.csv", index=False)
    (outputs / "model_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (outputs / "retrieval_recall_at_k.json").write_text(
        json.dumps(
            {
                "status": summary.get("status"),
                "recall_at_5": summary.get("recall_at_5"),
                "recall_at_10": summary.get("recall_at_10"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outputs / "mrr_at_10.json").write_text(
        json.dumps(
            {
                "bi_encoder_mrr_at_10": summary.get("bi_encoder_mrr_at_10"),
                "reranked_mrr_at_10": summary.get("reranked_mrr_at_10"),
                "mrr_improvement": summary.get("mrr_improvement"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outputs / "ndcg_at_10.json").write_text(
        json.dumps(
            {
                "bi_encoder_ndcg_at_10": summary.get("bi_encoder_ndcg_at_10"),
                "reranked_ndcg_at_10": summary.get("reranked_ndcg_at_10"),
                "ndcg_improvement": summary.get("ndcg_improvement"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if summary.get("status") == "completed":
        plot_metric_comparison(
            summary,
            outputs / "bi_encoder_vs_cross_encoder_comparison.png",
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
