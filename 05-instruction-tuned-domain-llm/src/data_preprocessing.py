"""Dataset loading, cleaning, validation, splitting, and statistics."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REQUIRED_FIELDS = ("instruction", "output", "category", "topic")
SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def save_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(record)
    cleaned["instruction"] = normalize_text(cleaned.get("instruction"))
    cleaned["input"] = normalize_text(cleaned.get("input"))
    output = cleaned.get("output", cleaned.get("response", cleaned.get("reference_answer", "")))
    cleaned["output"] = normalize_text(output)
    cleaned["response"] = cleaned["output"]
    cleaned["reference_answer"] = normalize_text(cleaned.get("reference_answer", cleaned["output"]))
    cleaned["category"] = normalize_text(cleaned.get("category", "uncategorized"))
    cleaned["difficulty"] = normalize_text(cleaned.get("difficulty", "unspecified"))
    cleaned["topic"] = normalize_text(cleaned.get("topic", "general"))
    cleaned["source"] = normalize_text(cleaned.get("source", "unspecified"))
    cleaned["id"] = normalize_text(cleaned.get("id"))
    cleaned["split"] = normalize_text(cleaned.get("split"))
    return cleaned


def validate_records(records: Iterable[dict[str, Any]], min_output_words: int = 4) -> dict[str, Any]:
    rows = [clean_record(row) for row in records]
    issues: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for index, row in enumerate(rows):
        key = row["instruction"].lower() + "\n" + row["input"].lower()
        seen[key] += 1
        missing = [field for field in REQUIRED_FIELDS if not row.get(field)]
        if missing:
            issues.append({"row": index, "type": "missing_required", "details": missing})
        if len(row["output"].split()) < min_output_words:
            issues.append({"row": index, "type": "short_output", "details": len(row["output"].split())})
        combined = " ".join(str(v) for v in row.values())
        if any(pattern.search(combined) for pattern in SENSITIVE_PATTERNS):
            issues.append({"row": index, "type": "possible_sensitive_data", "details": "review required"})
    duplicate_keys = {key for key, count in seen.items() if count > 1}
    for index, row in enumerate(rows):
        key = row["instruction"].lower() + "\n" + row["input"].lower()
        if key in duplicate_keys:
            issues.append({"row": index, "type": "duplicate_prompt", "details": key[:100]})
    return {
        "valid": len(issues) == 0,
        "row_count": len(rows),
        "issue_count": len(issues),
        "issues": issues,
    }


def assign_deterministic_splits(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign reproducible 80/10/10 splits when a split is missing."""
    output = []
    for index, original in enumerate(records):
        row = clean_record(original)
        if row.get("split") not in {"train", "validation", "test"}:
            mod = index % 10
            row["split"] = "test" if mod == 0 else ("validation" if mod == 1 else "train")
        output.append(row)
    return output


def dataset_statistics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [clean_record(row) for row in records]
    if not rows:
        return {"rows": 0}
    prompt_lengths = [len((r["instruction"] + " " + r["input"]).split()) for r in rows]
    response_lengths = [len(r["output"].split()) for r in rows]
    return {
        "rows": len(rows),
        "categories": dict(Counter(r["category"] for r in rows)),
        "topics": len(set(r["topic"] for r in rows)),
        "splits": dict(Counter(r["split"] for r in rows)),
        "average_prompt_words": round(sum(prompt_lengths) / len(prompt_lengths), 2),
        "average_response_words": round(sum(response_lengths) / len(response_lengths), 2),
    }
