"""Section-aware, sentence-conscious document chunking."""

from __future__ import annotations

import re
from typing import Any

from .document_loader import LoadedDocument, slugify
from .text_preprocessing import extract_sections, normalize_for_deduplication

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`])")


def _window_long_words(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def split_text(text: str, chunk_size_words: int = 180, overlap_words: int = 40) -> list[str]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_size_words")

    sentences = [item.strip() for item in _SENTENCE_BOUNDARY.split(text) if item.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if len(sentence_words) > chunk_size_words:
            if current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            chunks.extend(_window_long_words(sentence_words, chunk_size_words, overlap_words))
            continue

        if current and current_words + len(sentence_words) > chunk_size_words:
            completed = " ".join(current)
            chunks.append(completed)
            overlap = completed.split()[-overlap_words:] if overlap_words else []
            current = overlap + [sentence]
            current_words = len(overlap) + len(sentence_words)
        else:
            current.append(sentence)
            current_words += len(sentence_words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(
    document: LoadedDocument,
    chunk_size_words: int = 180,
    overlap_words: int = 40,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for section in extract_sections(document.text):
        for part_index, text in enumerate(split_text(section.text, chunk_size_words, overlap_words)):
            dedup_key = normalize_for_deduplication(text)
            if not dedup_key or dedup_key in seen:
                continue
            seen.add(dedup_key)
            chunk_id = f"{document.document_id}--{slugify(section.title)}--{part_index:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document.document_id,
                "project_name": document.project_name,
                "project_category": document.project_category,
                "source_file": document.source_file,
                "section_title": section.title,
                "text": text,
                "url_or_local_path": document.url_or_local_path,
                "tags": document.tags,
                "document_type": document.document_type,
                "created_from": document.created_from,
                "word_count": len(text.split()),
            })
    return chunks


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size_words: int = 180,
    overlap_words: int = 40,
) -> list[dict[str, Any]]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size_words, overlap_words)
    ]
