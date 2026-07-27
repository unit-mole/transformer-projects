from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class ClipEncoder:
    model_id: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("Install requirements-model.txt to use CLIP encoding") from exc
        self._torch = torch
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError("zero-length embedding encountered")
        return array / norm

    def encode_text(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        values = list(texts)
        rows: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(values), batch_size):
                inputs = self.processor(text=values[start:start + batch_size], return_tensors="pt", padding=True, truncation=True).to(self.device)
                features = self.model.get_text_features(**inputs)
                rows.append(features.detach().cpu().numpy())
        return self._normalize(np.concatenate(rows, axis=0))

    def encode_images(self, paths: Iterable[str | Path], batch_size: int = 16) -> np.ndarray:
        from PIL import Image
        values = [Path(path) for path in paths]
        rows: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(values), batch_size):
                images = [Image.open(path).convert("RGB") for path in values[start:start + batch_size]]
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**inputs)
                rows.append(features.detach().cpu().numpy())
        return self._normalize(np.concatenate(rows, axis=0))
