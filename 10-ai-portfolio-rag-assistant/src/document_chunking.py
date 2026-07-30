from __future__ import annotations

from dataclasses import dataclass, asdict
import re

@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    section: str
    text: str
    start_word: int
    end_word: int

    def to_dict(self) -> dict:
        return asdict(self)


def _sections(markdown: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^(#{1,6})\s+(.+)$", markdown, re.MULTILINE))
    if not matches:
        return [("Document", markdown)]
    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = markdown[:matches[0].start()].strip()
        if preamble:
            sections.append(("Preamble", preamble))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections.append((match.group(2).strip(), markdown[match.end():end].strip()))
    return sections


def chunk_markdown(document_id: str, markdown: str, size_words: int = 220, overlap_words: int = 50) -> list[TextChunk]:
    if size_words <= 0 or overlap_words < 0 or overlap_words >= size_words:
        raise ValueError("Require size_words > overlap_words >= 0.")
    chunks: list[TextChunk] = []
    sequence = 0
    for section, body in _sections(markdown):
        words = body.split()
        if not words:
            continue
        step = size_words - overlap_words
        for start in range(0, len(words), step):
            segment = words[start:start + size_words]
            if not segment:
                break
            sequence += 1
            chunks.append(TextChunk(
                chunk_id=f"{document_id}:chunk-{sequence:04d}",
                document_id=document_id,
                section=section,
                text=" ".join(segment),
                start_word=start,
                end_word=start + len(segment),
            ))
            if start + size_words >= len(words):
                break
    return chunks
