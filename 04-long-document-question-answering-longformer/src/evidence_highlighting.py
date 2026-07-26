from __future__ import annotations

import html
import re
from typing import Optional

from .schemas import TextChunk


def _case_insensitive_find(text: str, query: str) -> tuple[int, int] | tuple[None, None]:
    if not text or not query:
        return None, None
    match = re.search(re.escape(query.strip()), text, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None, None


def highlight_answer_in_paragraph(
    paragraph: Optional[TextChunk],
    answer: str,
    answer_start_in_document: Optional[int],
    answer_end_in_document: Optional[int],
) -> str:
    if paragraph is None:
        return (
            "<div class='evidence-box'><strong>Evidence unavailable.</strong> "
            "The predicted span could not be mapped to a document paragraph.</div>"
        )

    local_start: Optional[int] = None
    local_end: Optional[int] = None
    if (
        answer_start_in_document is not None
        and answer_end_in_document is not None
        and paragraph.start_char <= answer_start_in_document < paragraph.end_char
    ):
        local_start = answer_start_in_document - paragraph.start_char
        local_end = min(
            answer_end_in_document - paragraph.start_char,
            len(paragraph.text),
        )

    if local_start is None or local_end is None or local_end <= local_start:
        local_start, local_end = _case_insensitive_find(paragraph.text, answer)

    escaped_paragraph = html.escape(paragraph.text)
    if local_start is None or local_end is None:
        return (
            "<div class='evidence-box'><strong>Supporting paragraph "
            f"{paragraph.chunk_id}:</strong><br>{escaped_paragraph}<br><em>"
            "The answer text could not be confidently highlighted inside this "
            "paragraph.</em></div>"
        )

    before = html.escape(paragraph.text[:local_start])
    marked = html.escape(paragraph.text[local_start:local_end])
    after = html.escape(paragraph.text[local_end:])
    return (
        "<div class='evidence-box'><strong>Supporting paragraph "
        f"{paragraph.chunk_id}:</strong><br>{before}"
        f"<mark>{marked}</mark>{after}</div>"
    )
