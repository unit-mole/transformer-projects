"""Dataset loading, cleaning, validation, grouped splitting, and statistics."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REQUIRED_FIELDS = ("instruction", "output", "category", "topic")
SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
]


def normalize_text(value: Any) -> str:
    """Normalize single-line fields such as instructions, topics, and labels."""
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_multiline(value: Any) -> str:
    """Normalize generated/reference text while preserving code and paragraphs."""
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]
    # Collapse more than two consecutive blank lines without destroying code fences.
    output: list[str] = []
    blank = 0
    for line in lines:
        if line.strip():
            blank = 0
            output.append(line)
        else:
            blank += 1
            if blank <= 1:
                output.append("")
    return "\n".join(output).strip()


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
    cleaned["input"] = normalize_multiline(cleaned.get("input"))
    output = cleaned.get("output", cleaned.get("response", cleaned.get("reference_answer", "")))
    cleaned["output"] = normalize_multiline(output)
    cleaned["response"] = cleaned["output"]
    cleaned["reference_answer"] = normalize_multiline(cleaned.get("reference_answer", cleaned["output"]))
    cleaned["category"] = normalize_text(cleaned.get("category", "uncategorized"))
    cleaned["difficulty"] = normalize_text(cleaned.get("difficulty", "unspecified"))
    cleaned["topic"] = normalize_text(cleaned.get("topic", "general"))
    cleaned["topic_group"] = normalize_text(cleaned.get("topic_group", cleaned["topic"])).lower()
    cleaned["source"] = normalize_text(cleaned.get("source", "unspecified"))
    cleaned["id"] = normalize_text(cleaned.get("id"))
    cleaned["split"] = normalize_text(cleaned.get("split"))
    return cleaned


def validate_records(
    records: Iterable[dict[str, Any]],
    min_output_words: int = 10,
    max_output_words: int = 300,
    enforce_group_isolation: bool = True,
) -> dict[str, Any]:
    rows = [clean_record(row) for row in records]
    issues: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    groups: defaultdict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        key = row["instruction"].lower() + "\n" + row["input"].lower()
        seen[key] += 1
        missing = [field for field in REQUIRED_FIELDS if not row.get(field)]
        if missing:
            issues.append({"row": index, "type": "missing_required", "details": missing})
        output_words = len(re.findall(r"\b\w+\b", row["output"]))
        if output_words < min_output_words:
            issues.append({"row": index, "type": "short_output", "details": output_words})
        if output_words > max_output_words:
            issues.append({"row": index, "type": "long_output", "details": output_words})
        combined = " ".join(str(v) for v in row.values())
        if any(pattern.search(combined) for pattern in SENSITIVE_PATTERNS):
            issues.append({"row": index, "type": "possible_sensitive_data", "details": "review required"})
        if row["split"]:
            groups[row["topic_group"]].add(row["split"])
    duplicate_keys = {key for key, count in seen.items() if count > 1}
    for index, row in enumerate(rows):
        key = row["instruction"].lower() + "\n" + row["input"].lower()
        if key in duplicate_keys:
            issues.append({"row": index, "type": "duplicate_prompt", "details": key[:100]})
    if enforce_group_isolation:
        for group, splits in groups.items():
            if len(splits) > 1:
                issues.append({"type": "topic_group_leakage", "details": {"topic_group": group, "splits": sorted(splits)}})
    return {
        "valid": len(issues) == 0,
        "row_count": len(rows),
        "issue_count": len(issues),
        "issues": issues,
    }


def assign_grouped_splits(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Assign deterministic group-isolated train/validation/test splits."""
    rows = [clean_record(row) for row in records]
    groups = sorted(
        {row["topic_group"] for row in rows},
        key=lambda value: hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest(),
    )
    n_groups = len(groups)
    n_train = int(round(n_groups * train_ratio))
    n_validation = int(round(n_groups * validation_ratio))
    n_train = min(max(n_train, 1), max(n_groups - 2, 1))
    n_validation = min(max(n_validation, 1), max(n_groups - n_train - 1, 1))
    split_lookup = {}
    for index, group in enumerate(groups):
        split_lookup[group] = "train" if index < n_train else ("validation" if index < n_train + n_validation else "test")
    for row in rows:
        row["split"] = split_lookup[row["topic_group"]]
    return rows


def assign_deterministic_splits(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve complete valid splits; otherwise assign grouped deterministic splits."""
    rows = [clean_record(row) for row in records]
    if all(row.get("split") in {"train", "validation", "test"} for row in rows):
        return rows
    return assign_grouped_splits(rows)


def dataset_statistics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [clean_record(row) for row in records]
    if not rows:
        return {"rows": 0}
    prompt_lengths = [len(re.findall(r"\b\w+\b", r["instruction"] + " " + r["input"])) for r in rows]
    response_lengths = [len(re.findall(r"\b\w+\b", r["output"])) for r in rows]
    return {
        "rows": len(rows),
        "categories": dict(Counter(r["category"] for r in rows)),
        "topics": len(set(r["topic"] for r in rows)),
        "topic_groups": len(set(r["topic_group"] for r in rows)),
        "splits": dict(Counter(r["split"] for r in rows)),
        "average_prompt_words": round(sum(prompt_lengths) / len(prompt_lengths), 2),
        "average_response_words": round(sum(response_lengths) / len(response_lengths), 2),
        "min_response_words": min(response_lengths),
        "max_response_words": max(response_lengths),
    }
