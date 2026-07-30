from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
from typing import Iterable

SUPPORTED_NAMES = {
    "README.md",
    "MODEL_CARD.md",
    "DATASET_CARD.md",
    "README_HUGGINGFACE.md",
    "README_GITHUB_PAGES.md",
    "README_VERCEL.md",
    "README_CLOUDFLARE.md",
    "PROJECT_ROADMAP.md",
}


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    source_path: str
    source_file: str
    project_id: str
    repository: str
    category_hint: str
    checksum_sha256: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def discover_markdown_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*.md")
        if path.name in SUPPORTED_NAMES or path.name.startswith("README")
    )


def _infer_location(relative: Path) -> tuple[str, str, str]:
    parts = relative.parts
    if len(parts) >= 4:
        category_hint, repository = parts[0], parts[1]
        project_id = parts[-2]
    elif len(parts) >= 3:
        category_hint = parts[0]
        project_id = parts[-2]
        repository_map = {
            "ann": "ann-deep-learning-projects",
            "simple-rnn": "simple-rnn-projects",
            "lstm": "lstm-projects",
            "bilstm": "bi-directional-lstm-projects",
            "cnn": "cnn-projects",
            "transformer": "transformer-projects",
        }
        repository = repository_map.get(category_hint.lower(), category_hint)
    elif len(parts) >= 2:
        category_hint = parts[0]
        repository = parts[0]
        project_id = parts[-2]
    else:
        category_hint = "Portfolio"
        repository = "portfolio"
        project_id = relative.stem
    return category_hint, repository, project_id


def load_documents(root: Path, paths: Iterable[Path] | None = None) -> list[SourceDocument]:
    files = list(paths) if paths is not None else discover_markdown_files(root)
    documents: list[SourceDocument] = []
    seen_hashes: set[str] = set()

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        checksum = sha256(text.encode("utf-8")).hexdigest()
        if checksum in seen_hashes:
            continue
        seen_hashes.add(checksum)
        relative = path.relative_to(root)
        category_hint, repository, project_id = _infer_location(relative)
        documents.append(
            SourceDocument(
                document_id=f"{repository}:{project_id}:{path.name}:{checksum[:10]}",
                source_path=relative.as_posix(),
                source_file=path.name,
                project_id=project_id,
                repository=repository,
                category_hint=category_hint,
                checksum_sha256=checksum,
                text=text,
            )
        )
    return documents
