from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

from .schemas import TextChunk
from .text_preprocessing import count_words, normalize_text


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")


def split_paragraphs_with_offsets(text: str) -> List[TextChunk]:
    """Split text into paragraphs while preserving character offsets."""
    normalized = normalize_text(text)
    chunks: List[TextChunk] = []
    if not normalized:
        return chunks

    cursor = 0
    chunk_id = 0
    for match in _PARAGRAPH_BREAK.finditer(normalized):
        raw = normalized[cursor : match.start()]
        left_trim = len(raw) - len(raw.lstrip())
        right_text = raw.rstrip()
        if right_text:
            start = cursor + left_trim
            end = cursor + len(right_text)
            paragraph = normalized[start:end]
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=paragraph,
                    start_char=start,
                    end_char=end,
                    word_count=count_words(paragraph),
                )
            )
            chunk_id += 1
        cursor = match.end()

    raw = normalized[cursor:]
    left_trim = len(raw) - len(raw.lstrip())
    right_text = raw.rstrip()
    if right_text:
        start = cursor + left_trim
        end = cursor + len(right_text)
        paragraph = normalized[start:end]
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=paragraph,
                start_char=start,
                end_char=end,
                word_count=count_words(paragraph),
            )
        )

    # A document with line breaks but no blank lines still needs usable sections.
    if len(chunks) <= 1 and "\n" in normalized:
        line_chunks: List[TextChunk] = []
        cursor = 0
        for line in normalized.splitlines(keepends=True):
            stripped = line.strip()
            if stripped:
                start = cursor + len(line) - len(line.lstrip())
                end = start + len(stripped)
                line_chunks.append(
                    TextChunk(
                        chunk_id=len(line_chunks),
                        text=stripped,
                        start_char=start,
                        end_char=end,
                        word_count=count_words(stripped),
                    )
                )
            cursor += len(line)
        if len(line_chunks) > len(chunks):
            chunks = line_chunks

    if not chunks and normalized:
        chunks = [
            TextChunk(
                chunk_id=0,
                text=normalized,
                start_char=0,
                end_char=len(normalized),
                word_count=count_words(normalized),
            )
        ]
    return chunks


def chunk_text_by_words(
    text: str,
    max_words: int = 350,
    overlap_words: int = 60,
) -> List[TextChunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive.")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be non-negative and less than max_words.")

    normalized = normalize_text(text, preserve_paragraphs=False)
    if not normalized:
        return []

    matches = list(re.finditer(r"\S+", normalized))
    chunks: List[TextChunk] = []
    start_word = 0
    while start_word < len(matches):
        end_word = min(len(matches), start_word + max_words)
        start_char = matches[start_word].start()
        end_char = matches[end_word - 1].end()
        chunk_text = normalized[start_char:end_char]
        chunks.append(
            TextChunk(
                chunk_id=len(chunks),
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                word_count=end_word - start_word,
            )
        )
        if end_word == len(matches):
            break
        start_word = end_word - overlap_words
    return chunks


def locate_supporting_paragraph(
    text: str,
    answer_start: Optional[int],
    answer_end: Optional[int],
) -> tuple[Optional[TextChunk], List[TextChunk]]:
    paragraphs = split_paragraphs_with_offsets(text)
    if answer_start is None or answer_end is None:
        return None, paragraphs

    for paragraph in paragraphs:
        if paragraph.start_char <= answer_start < paragraph.end_char:
            return paragraph, paragraphs

    # If an answer starts exactly at a separator, choose the nearest paragraph.
    if paragraphs:
        nearest = min(
            paragraphs,
            key=lambda paragraph: min(
                abs(paragraph.start_char - answer_start),
                abs(paragraph.end_char - answer_start),
            ),
        )
        return nearest, paragraphs
    return None, paragraphs


def context_length_bucket(token_count: int) -> str:
    if token_count <= 512:
        return "0-512"
    if token_count <= 1024:
        return "513-1024"
    if token_count <= 2048:
        return "1025-2048"
    if token_count <= 4096:
        return "2049-4096"
    return "4097+"
