"""Semi-automated flags and a human-review template for hallucination analysis."""

from __future__ import annotations

import re
from typing import Any

ABSOLUTE_MARKERS = ("always", "never", "guaranteed", "100%", "perfectly")
HIGH_STAKES = ("legal", "medical", "financial", "immigration", "safety-critical")


def flag_hallucination_risks(instruction: str, response: str) -> dict[str, Any]:
    lower = response.lower()
    issue_types: list[str] = []
    if any(marker in lower for marker in ABSOLUTE_MARKERS):
        issue_types.append("overconfident_absolute_claim")
    if any(term in instruction.lower() for term in HIGH_STAKES):
        issue_types.append("outside_intended_scope")
    if re.search(r"\baccording to (a study|research|experts)\b", lower) and not re.search(r"https?://|\[[0-9]+\]", response):
        issue_types.append("unsupported_attribution")
    if "```" in response and "python" in instruction.lower():
        issue_types.append("code_requires_execution_review")
    severity = "none" if not issue_types else ("high" if "outside_intended_scope" in issue_types else "review")
    return {
        "hallucination_flag": bool(issue_types),
        "issue_types": issue_types,
        "severity": severity,
        "manual_note": "Verify definitions, numeric claims, code behavior, and omitted caveats against trusted sources.",
    }
