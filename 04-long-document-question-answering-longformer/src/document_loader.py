from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from .config import InferenceConfig
from .schemas import LoadedDocument
from .text_preprocessing import count_words, join_text_values, normalize_text


SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf"}
PREFERRED_TEXT_COLUMNS = (
    "text",
    "content",
    "document",
    "context",
    "paragraph",
    "body",
    "description",
)


class DocumentLoadingError(ValueError):
    """Raised when a document cannot be safely loaded."""


def _resolve_path(file_value: Any) -> Path:
    if isinstance(file_value, Path):
        return file_value
    if isinstance(file_value, str):
        return Path(file_value)
    name = getattr(file_value, "name", None)
    if name:
        return Path(name)
    raise DocumentLoadingError("The uploaded file path could not be resolved.")


def _check_file(path: Path, max_upload_mb: int) -> None:
    if not path.exists() or not path.is_file():
        raise DocumentLoadingError(f"Document not found: {path}")
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentLoadingError(
            f"Unsupported file type {extension or '<none>'}. Supported types: {allowed}."
        )
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_upload_mb:
        raise DocumentLoadingError(
            f"The file is {size_mb:.1f} MB. The configured maximum is {max_upload_mb} MB."
        )


def _load_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentLoadingError(
        "The text file encoding could not be decoded as UTF-8 or Windows-1252."
    )


def _load_csv(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise DocumentLoadingError("pandas is required to read CSV files.") from exc

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DocumentLoadingError(f"CSV parsing failed: {exc}") from exc

    if frame.empty:
        raise DocumentLoadingError("The CSV file contains no rows.")

    lower_to_original = {str(column).lower(): column for column in frame.columns}
    selected = [
        lower_to_original[name]
        for name in PREFERRED_TEXT_COLUMNS
        if name in lower_to_original
    ]
    if not selected:
        selected = [
            column
            for column in frame.columns
            if str(frame[column].dtype) in {"object", "string"}
        ]
    if not selected:
        raise DocumentLoadingError(
            "No text-like CSV column was found. Use a column such as text, content, "
            "document, context, paragraph, or body."
        )

    row_texts = []
    for _, row in frame[selected].iterrows():
        parts = []
        for column in selected:
            value = row[column]
            if value is None:
                continue
            try:
                if pd.isna(value):
                    continue
            except TypeError:
                pass
            cleaned = normalize_text(value, preserve_paragraphs=False)
            if cleaned:
                parts.append(f"{column}: {cleaned}" if len(selected) > 1 else cleaned)
        if parts:
            row_texts.append(" | ".join(parts))

    return "\n\n".join(row_texts), {
        "rows": int(len(frame)),
        "selected_text_columns": [str(column) for column in selected],
    }


def _load_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentLoadingError(
            "pypdf is required for PDF support. Install the project requirements."
        ) from exc

    try:
        reader = PdfReader(str(path))
        page_text = []
        empty_pages = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                page_text.append(text)
            else:
                empty_pages += 1
    except Exception as exc:
        raise DocumentLoadingError(f"PDF text extraction failed: {exc}") from exc

    if not page_text:
        raise DocumentLoadingError(
            "No selectable text was found in the PDF. Scanned PDFs require OCR, "
            "which is intentionally not included in the public CPU demo."
        )
    return "\n\n".join(page_text), {
        "pages": len(reader.pages),
        "pages_without_extractable_text": empty_pages,
    }


def _finalize_document(
    text: str,
    source_name: str,
    source_type: str,
    config: InferenceConfig,
    metadata: Optional[dict[str, Any]] = None,
) -> LoadedDocument:
    cleaned = normalize_text(text)
    if not cleaned:
        raise DocumentLoadingError("The selected document contains no readable text.")
    if len(cleaned) > config.max_document_characters:
        raise DocumentLoadingError(
            "The document contains "
            f"{len(cleaned):,} characters, exceeding the configured limit of "
            f"{config.max_document_characters:,}. Split the document or increase "
            "LONGDOCQA_MAX_DOCUMENT_CHARACTERS for local use."
        )
    return LoadedDocument(
        text=cleaned,
        source_name=source_name,
        source_type=source_type,
        character_count=len(cleaned),
        word_count=count_words(cleaned),
        metadata=metadata or {},
    )


def load_document_from_path(
    file_value: Any,
    config: Optional[InferenceConfig] = None,
) -> LoadedDocument:
    config = (config or InferenceConfig()).validate()
    path = _resolve_path(file_value)
    _check_file(path, config.max_upload_mb)

    extension = path.suffix.lower()
    metadata: dict[str, Any] = {
        "extension": extension,
        "size_bytes": path.stat().st_size,
    }
    if extension in {".txt", ".md"}:
        text = _load_text(path)
    elif extension == ".csv":
        text, csv_metadata = _load_csv(path)
        metadata.update(csv_metadata)
    elif extension == ".pdf":
        text, pdf_metadata = _load_pdf(path)
        metadata.update(pdf_metadata)
    else:  # protected by _check_file
        raise DocumentLoadingError(f"Unsupported extension: {extension}")

    return _finalize_document(
        text=text,
        source_name=path.name,
        source_type=f"uploaded_{extension.lstrip('.')}",
        config=config,
        metadata=metadata,
    )


def list_sample_documents(
    sample_directory: Optional[Union[str, Path]] = None,
) -> list[str]:
    directory = Path(sample_directory or InferenceConfig().sample_directory)
    if not directory.exists():
        return []
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_sample_document(
    sample_name: str,
    config: Optional[InferenceConfig] = None,
) -> LoadedDocument:
    config = (config or InferenceConfig()).validate()
    safe_name = Path(sample_name).name
    path = config.sample_directory / safe_name
    if path.parent.resolve() != config.sample_directory.resolve():
        raise DocumentLoadingError("Invalid sample document path.")
    document = load_document_from_path(path, config)
    document.source_type = "sample_document"
    return document


def load_document(
    uploaded_file: Any = None,
    manual_text: Optional[str] = None,
    sample_name: Optional[str] = None,
    config: Optional[InferenceConfig] = None,
) -> LoadedDocument:
    """Apply a deterministic priority: upload, manual text, then sample."""
    config = (config or InferenceConfig()).validate()

    if uploaded_file:
        return load_document_from_path(uploaded_file, config)
    if manual_text and normalize_text(manual_text):
        return _finalize_document(
            manual_text,
            source_name="manual_text",
            source_type="manual_text",
            config=config,
        )
    if sample_name:
        return load_sample_document(sample_name, config)
    raise DocumentLoadingError(
        "Provide an uploaded document, paste document text, or select a sample."
    )
