"""Dataset-level image safety and metadata helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from PIL import Image

from .image_preprocessing import SUPPORTED_SUFFIXES, load_rgb_image


def discover_images(root: str | Path) -> list[Path]:
    base = Path(root)
    return sorted(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def image_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_images(paths: Iterable[str | Path]) -> dict[str, list[str]]:
    valid, invalid = [], []
    for path in paths:
        try:
            load_rgb_image(path)
            valid.append(str(path))
        except ValueError:
            invalid.append(str(path))
    return {"valid": valid, "invalid": invalid}


def strip_exif(source: str | Path, destination: str | Path) -> Path:
    image = load_rgb_image(source)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format=Image.registered_extensions().get(output.suffix.lower(), "PNG"))
    return output
