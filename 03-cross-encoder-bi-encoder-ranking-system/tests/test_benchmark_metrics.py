import numpy as np

from src.benchmarking.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    paired_bootstrap_delta,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_retrieval_metrics_reward_better_ordering():
    relevance = {"A": 3, "B": 1}
    ideal = ["A", "B", "X"]
    weak = ["X", "B", "A"]

    assert recall_at_k(ideal, relevance, 2) == 1.0
    assert reciprocal_rank_at_k(ideal, relevance, 10) == 1.0
    assert reciprocal_rank_at_k(weak, relevance, 10) == 0.5
    assert ndcg_at_k(ideal, relevance, 3) > ndcg_at_k(weak, relevance, 3)
    assert average_precision_at_k(ideal, relevance, 100) > average_precision_at_k(
        weak, relevance, 100
    )


def test_paired_bootstrap_reports_positive_delta():
    before = np.array([0.0, 0.2, 0.4, 0.6])
    after = np.array([0.2, 0.4, 0.6, 0.8])
    result = paired_bootstrap_delta(before, after, samples=500, seed=7)

    assert result["mean_delta"] > 0
    assert result["ci_lower"] > 0
    assert result["probability_delta_positive"] == 1.0
