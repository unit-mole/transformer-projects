from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_HORIZONTAL_SPACE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: object, preserve_paragraphs: bool = True) -> str:
    """Normalize text while retaining paragraph boundaries by default."""
    if text is None:
        return ""

    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    value = _CONTROL_CHARS.sub(" ", value)

    if not preserve_paragraphs:
        return re.sub(r"\s+", " ", value).strip()

    lines: List[str] = []
    for line in value.split("\n"):
        cleaned = _HORIZONTAL_SPACE.sub(" ", line).strip()
        lines.append(cleaned)
    value = "\n".join(lines)
    value = _EXCESS_BLANK_LINES.sub("\n\n", value)
    return value.strip()


def nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in normalize_text(text).splitlines() if line.strip()]


def split_sentences(text: str) -> List[str]:
    cleaned = normalize_text(text, preserve_paragraphs=False)
    if not cleaned:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip()
    ]


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", normalize_text(text, False), flags=re.UNICODE))


def join_text_values(values: Iterable[object]) -> str:
    parts = [normalize_text(value) for value in values]
    return "\n\n".join(part for part in parts if part)
