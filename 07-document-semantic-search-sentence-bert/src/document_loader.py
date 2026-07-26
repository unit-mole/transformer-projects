"""Load public Markdown and text documents with portfolio metadata."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}


@dataclass(slots=True)
class LoadedDocument:
    document_id: str
    project_name: str
    project_category: str
    source_file: str
    document_type: str
    text: str
    url_or_local_path: str
    tags: list[str]
    created_from: str
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "document"


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse simple YAML-like frontmatter without requiring PyYAML."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    try:
        end_index = lines[1:].index("---") + 1
    except ValueError:
        return {}, text

    metadata: dict[str, Any] = {}
    for line in lines[1:end_index]:
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _parse_scalar(value)
    body = "\n".join(lines[end_index + 1 :]).lstrip()
    return metadata, body


def infer_document_type(filename: str) -> str:
    upper = filename.upper()
    if "MODEL_CARD" in upper:
        return "model_card"
    if "DATASET_CARD" in upper:
        return "dataset_card"
    if "DEPLOY" in upper or "GITHUB_PAGES" in upper:
        return "deployment_guide"
    if "README" in upper:
        return "project_readme"
    return "knowledge_note"


def first_markdown_heading(text: str) -> str | None:
    match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def load_document(path: Path, root: Path | None = None) -> LoadedDocument:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document extension: {path.suffix}")
    raw = path.read_text(encoding="utf-8")
    metadata, body = parse_frontmatter(raw)
    if not body.strip():
        raise ValueError(f"Document is empty after frontmatter removal: {path}")

    relative = path.relative_to(root) if root and path.is_relative_to(root) else path.name
    heading = first_markdown_heading(body)
    document_id = slugify(str(metadata.get("document_id") or path.stem))
    tags_value = metadata.get("tags", [])
    tags = tags_value if isinstance(tags_value, list) else [str(tags_value)]
    content_hash = hashlib.sha256(body.strip().encode("utf-8")).hexdigest()

    return LoadedDocument(
        document_id=document_id,
        project_name=str(metadata.get("project_name") or heading or path.stem.replace("-", " ").title()),
        project_category=str(metadata.get("project_category") or "Machine Learning Portfolio"),
        source_file=str(relative).replace("\\", "/"),
        document_type=str(metadata.get("document_type") or infer_document_type(path.name)),
        text=body.strip(),
        url_or_local_path=str(metadata.get("url") or relative).replace("\\", "/"),
        tags=[str(tag).strip() for tag in tags if str(tag).strip()],
        created_from=str(metadata.get("created_from") or "public synthetic portfolio documentation"),
        content_hash=content_hash,
    )


def iter_document_paths(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_documents(input_dir: Path) -> list[LoadedDocument]:
    documents: list[LoadedDocument] = []
    seen_hashes: set[str] = set()
    for path in iter_document_paths(input_dir):
        try:
            document = load_document(path, root=input_dir)
        except ValueError as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if document.content_hash in seen_hashes:
            print(f"Skipping duplicate document: {path}")
            continue
        seen_hashes.add(document.content_hash)
        documents.append(document)
    return documents
