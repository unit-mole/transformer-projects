from src.instruction_adherence import score_instruction_adherence


def test_adherence_returns_documented_score():
    result = score_instruction_adherence("Explain precision", "Precision measures how many predicted positives are correct.")
    assert 0 <= result["adherence_score"] <= 1
    assert result["answered"]
