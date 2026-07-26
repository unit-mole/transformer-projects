from __future__ import annotations

from src.document_chunking import (
    chunk_text_by_words,
    context_length_bucket,
    locate_supporting_paragraph,
    split_paragraphs_with_offsets,
)


def test_paragraph_offsets_map_to_original_normalized_text() -> None:
    text = "Paragraph one.\n\nParagraph two contains the answer."
    paragraphs = split_paragraphs_with_offsets(text)

    assert len(paragraphs) == 2
    second = paragraphs[1]
    assert text[second.start_char : second.end_char] == second.text

    answer_start = text.index("the answer")
    paragraph, _ = locate_supporting_paragraph(
        text,
        answer_start,
        answer_start + len("the answer"),
    )
    assert paragraph is not None
    assert paragraph.chunk_id == 1


def test_word_chunk_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(30))
    chunks = chunk_text_by_words(text, max_words=10, overlap_words=2)

    assert len(chunks) == 4
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


def test_context_length_buckets() -> None:
    assert context_length_bucket(512) == "0-512"
    assert context_length_bucket(1024) == "513-1024"
    assert context_length_bucket(2048) == "1025-2048"
    assert context_length_bucket(4096) == "2049-4096"
    assert context_length_bucket(4097) == "4097+"
