from __future__ import annotations

import re
from collections.abc import Iterable


def clean_text(value: str, *, field_name: str = "text", max_length: int = 500) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = re.sub(r"\s+", " ", value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    if len(cleaned) > max_length:
        raise ValueError(f"{field_name} must be {max_length} characters or fewer")
    return cleaned


def parse_candidate_labels(value: str | Iterable[str], *, max_labels: int = 24) -> list[str]:
    if isinstance(value, str):
        raw = re.split(r"[,;\n]", value)
    else:
        raw = list(value)
    labels: list[str] = []
    seen: set[str] = set()
    for item in raw:
        candidate = re.sub(r"\s+", " ", str(item)).strip()
        if not candidate:
            continue
        label = clean_text(candidate, field_name="candidate label", max_length=80).lower()
        if label not in seen:
            labels.append(label)
            seen.add(label)
    if len(labels) < 2:
        raise ValueError("at least two candidate labels are required")
    if len(labels) > max_labels:
        raise ValueError(f"use {max_labels} candidate labels or fewer")
    return labels


def create_label_prompts(labels: Iterable[str], template: str = "a photo of a {label}") -> list[str]:
    if "{label}" not in template:
        raise ValueError("prompt template must contain {label}")
    return [template.format(label=clean_text(label, field_name="label", max_length=80)) for label in labels]
