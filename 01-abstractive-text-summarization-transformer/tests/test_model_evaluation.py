from src.model_evaluation import compute_rouge


def test_rouge_identical_summary_is_one() -> None:
    metrics = compute_rouge(["the quality team improved response time"], ["the quality team improved response time"])
    assert metrics["rouge1"] == 1.0
    assert metrics["rouge2"] == 1.0
    assert metrics["rougeL"] == 1.0
