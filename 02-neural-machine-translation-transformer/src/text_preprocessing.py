from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

HTML_TAG_RE = re.compile(r"<[^>]+>")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: Any, *, max_characters: int = 5000) -> str:
    """Clean text without removing meaningful English or Devanagari content."""
    if value is None:
        return ""

    text = str(value)
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00A0", " ")
    text = CONTROL_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    if len(text) > max_characters:
        raise ValueError(
            f"Input contains {len(text)} characters; maximum is {max_characters}."
        )
    return text


def clean_parallel_pair(
    english: Any,
    hindi: Any,
    *,
    max_characters: int = 5000,
) -> tuple[str, str]:
    return (
        clean_text(english, max_characters=max_characters),
        clean_text(hindi, max_characters=max_characters),
    )


def approximate_word_count(text: Any) -> int:
    cleaned = clean_text(text)
    return len(cleaned.split()) if cleaned else 0
