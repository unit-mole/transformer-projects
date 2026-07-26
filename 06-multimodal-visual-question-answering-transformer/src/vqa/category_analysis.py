from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

def summarize_by_category(records: Iterable[Mapping[str, object]]) -> list[dict]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        category = str(record.get("category", record.get("question_type", "other")))
        score = float(record.get("vqa_score", record.get("correct", 0.0)))
        buckets[category].append(score)
    return [
        {
            "category": category,
            "total_questions": len(scores),
            "correct_equivalent": round(sum(scores), 4),
            "accuracy": round(sum(scores) / len(scores), 6),
        }
        for category, scores in sorted(buckets.items())
    ]
