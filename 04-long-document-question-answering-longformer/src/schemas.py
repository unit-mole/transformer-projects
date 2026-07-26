from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LoadedDocument:
    text: str
    source_name: str
    source_type: str
    character_count: int
    word_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TextChunk:
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    word_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpanCandidate:
    answer: str
    start_char: int
    end_char: int
    raw_score: float
    confidence_proxy: float
    feature_index: int
    start_token: int
    end_token: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QAResult:
    answer: str
    confidence_proxy: float
    confidence_label: str
    supporting_paragraph: str
    highlighted_evidence_html: str
    paragraph_index: Optional[int]
    answer_start_char: Optional[int]
    answer_end_char: Optional[int]
    model_id: str
    model_max_length: int
    requested_max_length: int
    window_count: int
    document_character_count: int
    document_word_count: int
    document_token_count: Optional[int]
    latency_seconds: float
    source_name: str
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
