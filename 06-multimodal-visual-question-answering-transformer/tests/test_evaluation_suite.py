from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_evaluation_suite_is_balanced_and_complete() -> None:
    root_path = PROJECT_ROOT / "data" / "evaluation" / "vqa_evaluation_60.json"
    space_path = PROJECT_ROOT / "space" / "evaluation" / "vqa_evaluation_60.json"

    root_records = json.loads(root_path.read_text(encoding="utf-8"))
    space_records = json.loads(space_path.read_text(encoding="utf-8"))

    assert root_records == space_records
    assert len(root_records) == 60
    assert Counter(record["category"] for record in root_records) == Counter(
        {
            "color": 10,
            "object": 10,
            "counting": 10,
            "yes_no": 10,
            "action_scene": 10,
            "spatial": 10,
        }
    )

    for record in root_records:
        assert record["accepted_answers"]
        relative = record["image"].replace("./evaluation/", "")
        assert (PROJECT_ROOT / "data" / "evaluation" / relative).is_file()
        assert (PROJECT_ROOT / "space" / "evaluation" / relative).is_file()
