import pytest
from src.class_mapping import build_mappings, validate_class_names


def test_build_mappings():
    id2label, label2id = build_mappings(["cat", "dog"])
    assert id2label == {0: "cat", 1: "dog"}
    assert label2id == {"cat": 0, "dog": 1}


def test_duplicate_names_fail():
    with pytest.raises(ValueError):
        validate_class_names(["cat", "cat"])
