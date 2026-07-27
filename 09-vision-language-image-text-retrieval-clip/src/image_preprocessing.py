from __future__ import annotations

from pathlib import Path
from PIL import Image, UnidentifiedImageError

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def load_image_rgb(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported image format: {image_path.suffix}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB").copy()
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"could not load image: {image_path}") from exc


def save_exif_free(image: Image.Image, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(target, format="PNG", optimize=True)
    return target


def validate_dimensions(image: Image.Image, *, minimum: int = 32, maximum: int = 8192) -> None:
    width, height = image.size
    if min(width, height) < minimum:
        raise ValueError(f"image dimensions must be at least {minimum} pixels")
    if max(width, height) > maximum:
        raise ValueError(f"image dimensions must not exceed {maximum} pixels")
