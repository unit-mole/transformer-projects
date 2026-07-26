from src.data_preprocessing import assign_deterministic_splits, clean_record, validate_records


def test_clean_record_maps_response_to_output():
    row = clean_record({"instruction": " Explain ML ", "response": " A useful answer ", "category": "concept", "topic": "ml"})
    assert row["instruction"] == "Explain ML"
    assert row["output"] == "A useful answer"


def test_split_assignment_is_reproducible():
    rows = [{"instruction": f"Question {i}", "output": "A valid educational response", "category": "c", "topic": "t"} for i in range(12)]
    first = assign_deterministic_splits(rows)
    second = assign_deterministic_splits(rows)
    assert [r["split"] for r in first] == [r["split"] for r in second]


def test_validation_detects_duplicate_prompt():
    rows = [
        {"instruction": "Same", "output": "A complete response here", "category": "c", "topic": "t"},
        {"instruction": "Same", "output": "Another complete response", "category": "c", "topic": "t"},
    ]
    report = validate_records(rows)
    assert not report["valid"]
    assert any(issue["type"] == "duplicate_prompt" for issue in report["issues"])
