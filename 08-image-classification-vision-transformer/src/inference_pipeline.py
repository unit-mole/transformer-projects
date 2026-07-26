"""Reusable Python inference helpers."""
from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def get_top_k_predictions(probabilities: np.ndarray, id2label: dict[int, str], k: int = 5) -> list[dict]:
    values = np.asarray(probabilities).reshape(-1)
    k = max(1, min(int(k), values.size))
    indices = np.argsort(values)[::-1][:k]
    return [{"class_id": int(i), "label": id2label[int(i)], "score": float(values[i])} for i in indices]


def predict_class(model: Any, processor: Any, image: Image.Image, top_k: int = 5) -> dict:
    import torch
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    start = perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    latency_ms = (perf_counter() - start) * 1000.0
    probs = outputs.logits.softmax(dim=-1)[0].cpu().numpy()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    predictions = get_top_k_predictions(probs, id2label, top_k)
    return {"prediction": predictions[0], "top_k": predictions, "latency_ms": latency_ms}
