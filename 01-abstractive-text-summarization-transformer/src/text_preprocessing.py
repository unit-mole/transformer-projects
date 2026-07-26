from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def clean_text(value: Any, *, remove_html: bool = True) -> str:
    """Clean text without deleting Unicode names, symbols, dates, or numbers."""
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    if remove_html:
        text = _HTML_TAG_RE.sub(" ", text)
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def split_sentences(text: Any) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]


def word_count(text: Any) -> int:
    return len(re.findall(r"\b[\w'-]+\b", clean_text(text), flags=re.UNICODE))


def validate_article(text: Any, *, min_words: int = 25) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Please provide an article or long text.")
    count = word_count(cleaned)
    if count < min_words:
        raise ValueError(f"Please provide at least {min_words} words; received {count}.")
    return cleaned
