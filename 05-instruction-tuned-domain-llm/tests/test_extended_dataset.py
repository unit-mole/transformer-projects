from pathlib import Path

from scripts.build_extended_dataset import build_dataset
from src.data_preprocessing import load_jsonl, validate_records


def test_extended_dataset_is_large_and_group_isolated(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    dataset_path = tmp_path / "extended.jsonl"
    evaluation_path = tmp_path / "evaluation.jsonl"
    result = build_dataset(
        project / "data" / "ml_ds_instruction_dataset.jsonl",
        dataset_path,
        evaluation_path,
        tmp_path / "stats.json",
        tmp_path / "validation.json",
    )
    rows = load_jsonl(dataset_path)
    assert result["validation"]["valid"]
    assert len(rows) >= 350
    assert len(load_jsonl(evaluation_path)) >= 25
    assert validate_records(rows, enforce_group_isolation=True)["valid"]
