"""Markdown-aware preprocessing that preserves retrieval-critical terms."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class Section:
    title: str
    text: str
    level: int


def clean_markdown(text: str) -> str:
    """Normalize Markdown while preserving headings, code text, model names, and metrics."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^```[^\n]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^~~~[^\n]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*\|\s*", " | ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_sections(text: str) -> list[Section]:
    """Split Markdown into heading-aware sections."""
    cleaned = clean_markdown(text)
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$", flags=re.MULTILINE)
    matches = list(heading_pattern.finditer(cleaned))
    if not matches:
        return [Section(title="Document", text=cleaned, level=1)] if cleaned else []

    sections: list[Section] = []
    preamble = cleaned[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(title="Overview", text=preamble, level=1))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        body = cleaned[start:end].strip()
        title = match.group(2).strip()
        if body:
            sections.append(Section(title=title, text=body, level=len(match.group(1))))
    return sections


def normalize_for_deduplication(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()
