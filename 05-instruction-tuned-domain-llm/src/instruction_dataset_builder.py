"""Build a Hugging Face DatasetDict from validated JSONL records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_preprocessing import assign_deterministic_splits, load_jsonl, validate_records
from .prompt_templates import format_prompt


def build_dataset_dict(path: str | Path):
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to build DatasetDict objects.") from exc

    rows = assign_deterministic_splits(load_jsonl(path))
    report = validate_records(rows)
    if not report["valid"]:
        raise ValueError(f"Dataset validation failed: {report['issues'][:5]}")

    prepared: list[dict[str, Any]] = []
    for row in rows:
        prepared.append({
            **row,
            "prompt": format_prompt(row["instruction"], row.get("input", "")),
            "target": row["output"],
        })

    return DatasetDict({
        split: Dataset.from_list([row for row in prepared if row["split"] == split])
        for split in ("train", "validation", "test")
    })
