from __future__ import annotations

from src.evidence_highlighting import highlight_answer_in_paragraph
from src.schemas import TextChunk


def test_highlight_answer_uses_document_offsets() -> None:
    paragraph = TextChunk(
        chunk_id=2,
        text="Priya Raman was assigned as the CAPA owner.",
        start_char=100,
        end_char=144,
        word_count=8,
    )
    html = highlight_answer_in_paragraph(
        paragraph=paragraph,
        answer="Priya Raman",
        answer_start_in_document=100,
        answer_end_in_document=111,
    )

    assert "<mark>Priya Raman</mark>" in html
    assert "Supporting paragraph 2" in html


def test_missing_paragraph_returns_honest_message() -> None:
    html = highlight_answer_in_paragraph(None, "answer", None, None)
    assert "Evidence unavailable" in html
