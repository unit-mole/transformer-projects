"""Image loading and deterministic preprocessing shared by tests/evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    center_crop: bool = False


def load_rgb_image(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")
    try:
        with Image.open(image_path) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode image: {image_path}") from exc


def resize_image(image: Image.Image, size: Sequence[int], center_crop: bool = False) -> Image.Image:
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    image = image.convert("RGB")
    if center_crop:
        return ImageOps.fit(image, (width, height), method=Image.Resampling.BICUBIC)
    return image.resize((width, height), Image.Resampling.BICUBIC)


def image_to_numpy(image: Image.Image, config: PreprocessingConfig = PreprocessingConfig()) -> np.ndarray:
    resized = resize_image(image, config.image_size, config.center_crop)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(config.mean, dtype=np.float32)
    std = np.asarray(config.std, dtype=np.float32)
    if np.any(std == 0):
        raise ValueError("Normalization standard deviation cannot contain zero.")
    normalized = (array - mean) / std
    return np.transpose(normalized, (2, 0, 1))[None, ...]


def preprocess_image(path: str | Path, config: PreprocessingConfig = PreprocessingConfig()) -> np.ndarray:
    return image_to_numpy(load_rgb_image(path), config)
