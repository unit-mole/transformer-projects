import pytest
from vqa.question_preprocessing import classify_question_type, preprocess_question

def test_question_is_trimmed_and_normalized():
    assert preprocess_question("  What   color is it  ") == "What color is it?"

def test_empty_question_rejected():
    with pytest.raises(ValueError):
        preprocess_question("   ")

def test_question_categories():
    assert classify_question_type("How many cars are visible?") == "number"
    assert classify_question_type("What color is the vehicle?") == "color"
