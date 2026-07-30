from __future__ import annotations

from collections import Counter

from src.dataset_expansion import assign_stratified_splits, remove_near_duplicates


def _record(index: int, category: str) -> dict[str, object]:
    return {
        "id": f"r{index}",
        "instruction": f"Explain unique concept number {index} for machine learning.",
        "input": "",
        "output": "This is a sufficiently detailed educational response about a machine learning concept and its limitation.",
        "category": category,
        "difficulty": "intermediate",
        "topic": "testing",
        "source": "test",
        "split": "train",
    }


def test_assign_stratified_splits_creates_all_splits() -> None:
    records = [_record(i, "Concept explanation" if i < 20 else "Metric explanation") for i in range(40)]
    assigned = assign_stratified_splits(records, seed=42)
    counts = Counter(str(row["split"]) for row in assigned)
    assert counts["train"] > counts["validation"] > 0
    assert counts["test"] > 0
    assert len(assigned) == 40


def test_remove_near_duplicates_removes_identical_instruction() -> None:
    records = [_record(1, "Concept explanation"), _record(2, "Concept explanation")]
    records[1]["instruction"] = records[0]["instruction"]
    kept, removed = remove_near_duplicates(records, threshold=0.8)
    assert len(kept) == 1
    assert len(removed) == 1
