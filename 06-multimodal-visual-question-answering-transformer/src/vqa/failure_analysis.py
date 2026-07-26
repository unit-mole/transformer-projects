from __future__ import annotations

from typing import Mapping

def assign_failure_type(record: Mapping[str, object]) -> str:
    question_type = str(record.get("question_type", "other")).lower()
    expected = str(record.get("answer", "")).strip().lower()
    predicted = str(record.get("prediction", "")).strip().lower()

    if expected == predicted:
        return "correct"
    if question_type == "number":
        return "wrong_count"
    if question_type == "color":
        return "wrong_color"
    if question_type == "yes_no":
        return "wrong_yes_no"
    if question_type == "spatial":
        return "spatial_reasoning"
    if question_type == "action":
        return "wrong_action"
    return "wrong_object_or_other"

def create_failure_rows(records):
    return [{**dict(record), "failure_type": assign_failure_type(record)} for record in records]
