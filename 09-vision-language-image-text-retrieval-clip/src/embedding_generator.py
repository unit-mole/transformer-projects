from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def save_browser_embeddings(image_ids: Iterable[str], embeddings: np.ndarray, output_path: str | Path, *, model_id: str) -> Path:
    ids = list(image_ids)
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or len(ids) != array.shape[0]:
        raise ValueError("image_ids and embeddings must have matching rows")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embeddings contain a zero vector")
    array = array / norms
    payload = {
        "model_id": model_id,
        "dtype": "float32-json",
        "dimension": int(array.shape[1]),
        "normalized": True,
        "generated": True,
        "vectors": [
            {"image_id": image_id, "embedding": [round(float(value), 8) for value in vector]}
            for image_id, vector in zip(ids, array, strict=True)
        ],
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
