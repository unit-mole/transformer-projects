"""Export processed artifacts to the static GitHub Pages app."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REQUIRED_FILES = (
    "corpus.json",
    "document_chunks.json",
    "embeddings.json",
    "metadata.json",
    "evaluation_queries.json",
)


def export_browser_data(processed_dir: Path, web_data_dir: Path) -> list[Path]:
    web_data_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for filename in REQUIRED_FILES:
        source = processed_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Required processed artifact is missing: {source}")
        with source.open(encoding="utf-8") as handle:
            json.load(handle)
        destination = web_data_dir / filename
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied
