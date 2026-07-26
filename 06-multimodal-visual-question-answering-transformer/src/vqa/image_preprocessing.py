from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Union

from PIL import Image, UnidentifiedImageError

ImageInput = Union[str, Path, bytes, bytearray, BinaryIO, Image.Image]

def load_and_validate_image(
    source: ImageInput,
    *,
    max_megapixels: float = 25.0,
    remove_metadata: bool = True,
) -> Image.Image:
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(BytesIO(source))
        else:
            image = Image.open(source)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Unsupported, corrupt, or unreadable image.") from exc

    megapixels = (image.width * image.height) / 1_000_000
    if megapixels > max_megapixels:
        raise ValueError(
            f"Image is too large ({megapixels:.1f} MP). "
            f"Maximum supported size is {max_megapixels:.1f} MP."
        )

    rgb = image.convert("RGB")
    if remove_metadata:
        clean = Image.new("RGB", rgb.size)
        clean.paste(rgb)
        rgb = clean
    return rgb

def image_summary(image: Image.Image) -> dict:
    return {
        "width": int(image.width),
        "height": int(image.height),
        "mode": image.mode,
        "megapixels": round(image.width * image.height / 1_000_000, 3),
    }
