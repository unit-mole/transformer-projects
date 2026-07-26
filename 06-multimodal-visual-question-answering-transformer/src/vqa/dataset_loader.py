from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

REQUIRED_COLUMNS = {"image_path", "question", "answer"}

def load_vqa_csv(path: str | Path) -> list[dict]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    for row in rows:
        if row.get("answers", "").strip().startswith("["):
            try:
                row["answers"] = json.loads(row["answers"])
            except json.JSONDecodeError:
                pass
    return rows
