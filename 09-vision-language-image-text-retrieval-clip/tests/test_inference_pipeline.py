import numpy as np
from src.model_evaluation import evaluate_recall
from src.similarity_analysis import summarize_similarity


def test_recall_metrics():
    metrics = evaluate_recall([["a", "b"], ["c", "d"]], [{"a"}, {"d"}], ks=(1, 2))
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_2"] == 1.0


def test_similarity_summary():
    summary = summarize_similarity([0.2, 0.5, 0.4])
    assert np.isclose(summary["maximum"], 0.5)
    assert summary["top_margin"] > 0
