from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_gallery(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    images = payload.get("images") if isinstance(payload, dict) else payload
    if not isinstance(images, list) or not images:
        raise ValueError("gallery JSON must contain a non-empty images list")
    required = {"image_id", "image_path", "caption", "category", "tags"}
    ids: set[str] = set()
    for index, item in enumerate(images):
        missing = required - set(item)
        if missing:
            raise ValueError(f"gallery item {index} is missing: {sorted(missing)}")
        if item["image_id"] in ids:
            raise ValueError(f"duplicate image_id: {item['image_id']}")
        ids.add(item["image_id"])
    return images
