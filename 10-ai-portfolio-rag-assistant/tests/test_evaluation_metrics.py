from src.evaluation_metrics import evaluate_ranking, summarize_query_metrics


def test_recall_counts_all_expected_projects_not_just_one_hit():
    row = evaluate_ranking(
        question_id="q",
        relevant_ids=["a", "b", "c", "d", "e", "f"],
        retrieved_ids=["a", "b", "x", "y", "z"],
        k=5,
    )
    assert row.hit_rate == 1.0
    assert row.precision == 0.4
    assert row.recall == 0.333333
    assert row.reciprocal_rank == 1.0


def test_duplicate_project_chunks_do_not_inflate_metrics():
    row = evaluate_ranking(
        question_id="q",
        relevant_ids=["a", "b"],
        retrieved_ids=["a", "a", "a", "b"],
        k=3,
    )
    assert row.retrieved_ids == ["a", "b"]
    assert row.recall == 1.0


def test_summary_averages_query_metrics():
    rows = [
        evaluate_ranking("q1", ["a"], ["a"], 1),
        evaluate_ranking("q2", ["b"], ["x"], 1),
    ]
    summary = summarize_query_metrics(rows)
    assert summary["hit_rate"] == 0.5
    assert summary["recall"] == 0.5
