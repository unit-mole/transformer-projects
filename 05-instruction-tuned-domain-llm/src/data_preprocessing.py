"""Dataset loading, validation, cleaning, and split utilities."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REQUIRED_FIELDS = ("instruction", "input", "output", "category", "difficulty", "topic", "source")
SPLITS = {"train", "validation", "test"}


@dataclass
class ValidationReport:
    total_records: int
    valid_records: int
    removed_records: int
    empty_instructions: int
    empty_outputs: int
    duplicate_instructions: int
    too_short_outputs: int
    too_long_outputs: int
    possible_pii: int
    confidential_references: int

    def to_dict(self) -> Dict[str, int]:
        return self.__dict__.copy()


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\u00a0", " ")).strip()


def load_jsonl(path: str | Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return records


def save_jsonl(records: Iterable[Dict[str, object]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _possible_pii(text: str) -> bool:
    email = bool(re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text))
    phone = bool(re.search(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", text))
    return email or phone


def validate_and_clean_records(
    records: Sequence[Dict[str, object]],
    *,
    min_output_words: int = 6,
    max_output_words: int = 350,
) -> tuple[List[Dict[str, object]], ValidationReport]:
    cleaned: List[Dict[str, object]] = []
    seen_instructions: set[str] = set()
    counters = {
        "empty_instructions": 0,
        "empty_outputs": 0,
        "duplicate_instructions": 0,
        "too_short_outputs": 0,
        "too_long_outputs": 0,
        "possible_pii": 0,
        "confidential_references": 0,
    }

    confidential_terms = ("confidential", "proprietary company", "internal customer data", "trade secret")

    for index, raw in enumerate(records):
        item = {field: normalize_text(raw.get(field, "")) for field in REQUIRED_FIELDS}
        item["id"] = normalize_text(raw.get("id", f"record_{index:04d}"))
        item["split"] = normalize_text(raw.get("split", "train")).lower()
        if item["split"] not in SPLITS:
            item["split"] = "train"

        if not item["instruction"]:
            counters["empty_instructions"] += 1
            continue
        if not item["output"]:
            counters["empty_outputs"] += 1
            continue

        key = item["instruction"].lower()
        if key in seen_instructions:
            counters["duplicate_instructions"] += 1
            continue
        seen_instructions.add(key)

        output_words = len(item["output"].split())
        if output_words < min_output_words:
            counters["too_short_outputs"] += 1
            continue
        if output_words > max_output_words:
            counters["too_long_outputs"] += 1
            continue

        combined = " ".join([item["instruction"], item["input"], item["output"]])
        if _possible_pii(combined):
            counters["possible_pii"] += 1
            continue
        lowered = combined.lower().replace("non-confidential", "")
        if any(term in lowered for term in confidential_terms):
            counters["confidential_references"] += 1
            continue

        cleaned.append(item)

    report = ValidationReport(
        total_records=len(records),
        valid_records=len(cleaned),
        removed_records=len(records) - len(cleaned),
        **counters,
    )
    return cleaned, report


def split_records(records: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    return {split: [r for r in records if r.get("split") == split] for split in ("train", "validation", "test")}
