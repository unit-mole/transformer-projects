import pytest
from src.groundedness_evaluation import validate_score

def test_groundedness_score_range():
    validate_score(0.8)
    with pytest.raises(ValueError):
        validate_score(1.1)
