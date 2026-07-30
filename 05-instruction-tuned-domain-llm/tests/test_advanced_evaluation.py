from __future__ import annotations

from src.advanced_evaluation import bootstrap_mean_ci, response_quality_rubric


def test_bootstrap_ci_contains_sample_mean() -> None:
    result = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4], samples=200, seed=42)
    assert result["lower"] <= result["mean"] <= result["upper"]
    assert result["n"] == 4


def test_quality_rubric_rewards_caveat_and_topic() -> None:
    record = {
        "instruction": "Explain random forest and include one limitation.",
        "category": "Concept explanation",
        "topic": "random forest",
    }
    response = (
        "A random forest combines many decision trees trained on resampled data and random feature subsets. "
        "It can model nonlinear relationships and is a strong tabular baseline. However, feature importance can be biased "
        "and the ensemble is less interpretable than a single shallow tree."
    )
    result = response_quality_rubric(record, response)
    assert result["quality_rubric_score"] >= 0.8
