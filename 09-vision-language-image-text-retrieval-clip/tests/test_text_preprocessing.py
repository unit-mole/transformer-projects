import pytest
from src.text_preprocessing import clean_text, create_label_prompts, parse_candidate_labels


def test_clean_text_collapses_whitespace():
    assert clean_text("  red   car \n road ") == "red car road"


def test_clean_text_rejects_empty():
    with pytest.raises(ValueError):
        clean_text("   ")


def test_labels_are_unique_and_prompted():
    labels = parse_candidate_labels("Dog, cat, dog")
    assert labels == ["dog", "cat"]
    assert create_label_prompts(labels) == ["a photo of a dog", "a photo of a cat"]


def test_candidate_labels_ignore_empty_segments():
    assert parse_candidate_labels("dog, cat, ; ") == ["dog", "cat"]
