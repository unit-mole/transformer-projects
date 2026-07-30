from src.answer_evaluation import extract_citation_ids, split_claims
from src.local_generator import grounded_extractive_answer


def test_extract_citation_ids_is_unique_and_sorted():
    assert extract_citation_ids("Claim [S2] and [S1], repeated [S2].") == ["S1", "S2"]


def test_split_claims_ignores_portfolio_boilerplate():
    answer = (
        "Based on the indexed portfolio evidence:\n"
        "Project 01 uses a Transformer [S1].\n"
        "This answer is limited to the current corpus."
    )
    assert split_claims(answer) == ["Project 01 uses a Transformer [S1]."]


def test_extractive_generator_refuses_weak_retrieval():
    result = grounded_extractive_answer(
        "What is the user's salary?",
        [{"projectName": "Unrelated", "text": "No salary information.", "_retrievalScore": 0.05}],
        min_retrieval_score=0.20,
    )
    assert "could not find enough supporting information" in result.text.lower()
