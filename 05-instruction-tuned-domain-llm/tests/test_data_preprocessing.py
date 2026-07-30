from src.data_preprocessing import validate_and_clean_records


def test_validation_removes_empty_and_duplicates():
    records = [
        {"instruction": "Explain precision.", "input": "", "output": "Precision measures the correctness of positive predictions in classification.", "category": "Metric explanation", "difficulty": "beginner", "topic": "metrics", "source": "test", "split": "train"},
        {"instruction": "Explain precision.", "input": "", "output": "A duplicate answer with enough words for the validator to inspect.", "category": "Metric explanation", "difficulty": "beginner", "topic": "metrics", "source": "test", "split": "train"},
        {"instruction": "", "input": "", "output": "This record should be removed because the instruction is empty.", "category": "Metric explanation", "difficulty": "beginner", "topic": "metrics", "source": "test", "split": "train"},
    ]
    cleaned, report = validate_and_clean_records(records)
    assert len(cleaned) == 1
    assert report.duplicate_instructions == 1
    assert report.empty_instructions == 1
