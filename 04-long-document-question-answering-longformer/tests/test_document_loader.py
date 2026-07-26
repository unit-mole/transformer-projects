from __future__ import annotations

from pathlib import Path

import pytest

from src.config import InferenceConfig
from src.document_loader import (
    DocumentLoadingError,
    load_document_from_path,
)


def test_load_text_document(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
    config = InferenceConfig(sample_directory=tmp_path)

    document = load_document_from_path(path, config)

    assert document.source_name == "sample.txt"
    assert document.word_count == 4
    assert "Second paragraph" in document.text


def test_reject_empty_document(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n", encoding="utf-8")
    config = InferenceConfig(sample_directory=tmp_path)

    with pytest.raises(DocumentLoadingError, match="no readable text"):
        load_document_from_path(path, config)


def test_reject_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "sample.docx"
    path.write_bytes(b"not a real docx")
    config = InferenceConfig(sample_directory=tmp_path)

    with pytest.raises(DocumentLoadingError, match="Unsupported file type"):
        load_document_from_path(path, config)
