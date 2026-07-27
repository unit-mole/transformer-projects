from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset_loader import load_gallery
from .image_preprocessing import load_image_rgb, validate_dimensions


def validate_gallery_assets(gallery_path: str | Path, web_root: str | Path) -> dict[str, Any]:
    gallery = load_gallery(gallery_path)
    root = Path(web_root)
    categories: set[str] = set()
    for item in gallery:
        relative = str(item["image_path"]).removeprefix("./")
        image = load_image_rgb(root / relative)
        validate_dimensions(image)
        categories.add(item["category"])
    return {"images": len(gallery), "categories": len(categories), "valid": True}


def write_validation_report(report: dict[str, Any], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
