"""Semi-automated hallucination flags plus a manual-review schema."""
from __future__ import annotations

import re
from typing import Dict, List

ABSOLUTE_PATTERNS = (
    r"\balways\b", r"\bnever\b", r"\bguarantee(?:d|s)?\b", r"\b100 percent\b", r"\bperfect(?:ly)?\b"
)
UNVERIFIED_CITATION = re.compile(r"\b(?:according to|study|research)\b.*\b(?:19|20)\d{2}\b", re.IGNORECASE)


def analyze_hallucination_risk(prompt: str, response: str, reference: str = "") -> Dict[str, object]:
    flags: List[str] = []
    lower = response.lower()
    if any(re.search(pattern, lower) for pattern in ABSOLUTE_PATTERNS):
        flags.append("overconfident_absolute_claim")
    if UNVERIFIED_CITATION.search(response):
        flags.append("citation_or_study_claim_requires_verification")
    if re.search(r"\b(as an ai|i was trained|my database)\b", lower):
        flags.append("unsupported_model_self_claim")
    if len(response.split()) < 5:
        flags.append("answer_too_short_for_reliable_explanation")

    if reference:
        from .relevance_scoring import score_relevance
        similarity = float(score_relevance(reference, response)["combined_relevance"])
        if similarity < 0.08:
            flags.append("low_similarity_to_reference_review_required")
    else:
        similarity = None

    severity = "none" if not flags else ("high" if len(flags) >= 3 else "medium" if len(flags) == 2 else "low")
    return {
        "hallucination_flag": bool(flags),
        "issue_types": flags,
        "severity": severity,
        "reference_similarity": similarity,
        "manual_note": "Human factual review is required; heuristic flags are not proof of hallucination.",
        "corrected_explanation": "",
    }
