from __future__ import annotations

from dataclasses import dataclass, asdict

@dataclass
class CitationReview:
    question: str
    generated_answer: str
    citation_ids: list[str]
    citation_correctness_score: float | None = None
    unsupported_claims: list[str] | None = None
    reviewer_notes: str = "Pending human review"

    def to_dict(self) -> dict:
        return asdict(self)


def cited_ids(answer: str) -> list[str]:
    import re
    return sorted(set(re.findall(r"\[(S\d+)\]", answer)))
