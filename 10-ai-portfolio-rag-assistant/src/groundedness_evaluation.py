from __future__ import annotations

from dataclasses import dataclass, asdict

@dataclass
class GroundednessReview:
    question: str
    answer: str
    retrieved_sources: list[str]
    groundedness_score: float | None = None
    unsupported_claims: list[str] | None = None
    correction_needed: bool | None = None
    reviewer_notes: str = "Pending human review"

    def to_dict(self) -> dict:
        return asdict(self)


def validate_score(score: float | None) -> None:
    if score is not None and not 0 <= score <= 1:
        raise ValueError("Groundedness score must be between 0 and 1.")
