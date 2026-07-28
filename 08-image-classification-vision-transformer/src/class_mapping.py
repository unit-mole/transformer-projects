"""Class-label validation and serialization utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def validate_class_names(class_names: Iterable[str]) -> list[str]:
    names = [str(name).strip() for name in class_names]
    if not names:
        raise ValueError("At least one class name is required.")
    if any(not name for name in names):
        raise ValueError("Class names cannot be empty.")
    if len(names) != len(set(names)):
        raise ValueError("Class names must be unique.")
    return names


def build_mappings(class_names: Iterable[str]) -> tuple[dict[int, str], dict[str, int]]:
    names = validate_class_names(class_names)
    id2label = {index: name for index, name in enumerate(names)}
    label2id = {name: index for index, name in id2label.items()}
    return id2label, label2id


def save_class_names(class_names: Iterable[str], output_path: str | Path) -> Path:
    names = validate_class_names(class_names)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(names, indent=2), encoding="utf-8")
    return output


def load_class_names(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("classes", payload.get("class_names", []))
    return validate_class_names(payload)
