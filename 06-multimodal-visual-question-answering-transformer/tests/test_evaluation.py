from vqa.evaluation import evaluate_records, vqa_consensus_score

def test_vqa_consensus_score():
    assert vqa_consensus_score("cat", ["cat", "cat", "cat", "dog"]) == 1.0
    assert vqa_consensus_score("cat", ["cat", "dog", "bird"]) == 1 / 3

def test_evaluate_records():
    result = evaluate_records([
        {"prediction": "yes", "answers": ["yes"] * 10, "question_type": "yes_no"},
        {"prediction": "2", "answers": ["3"] * 10, "question_type": "number"},
    ])
    assert result["count"] == 2
    assert result["vqa_accuracy"] == 0.5
