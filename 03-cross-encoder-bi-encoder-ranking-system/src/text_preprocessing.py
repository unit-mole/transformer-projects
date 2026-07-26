from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    """Clean text without removing meaningful Unicode, numbers, or domain terms."""
    if value is None:
        return ""

    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def is_valid_text(value: Any, minimum_characters: int = 3) -> bool:
    return len(clean_text(value)) >= minimum_characters


def combine_title_and_document(title: Any, document: Any) -> str:
    title_text = clean_text(title)
    document_text = clean_text(document)
    if title_text and document_text:
        return f"{title_text}. {document_text}"
    return title_text or document_text
