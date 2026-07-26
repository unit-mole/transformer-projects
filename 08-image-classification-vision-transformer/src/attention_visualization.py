"""Attention rollout for compatible Hugging Face ViT/DeiT models.

This module creates genuine attention-derived maps only when the model returns
attention tensors. It intentionally does not fabricate attention from logits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def attention_rollout(attentions: Iterable, discard_ratio: float = 0.0, head_fusion: str = "mean") -> np.ndarray:
    matrices = []
    for layer in attentions:
        array = layer.detach().cpu().numpy()[0]
        if head_fusion == "mean":
            fused = array.mean(axis=0)
        elif head_fusion == "max":
            fused = array.max(axis=0)
        elif head_fusion == "min":
            fused = array.min(axis=0)
        else:
            raise ValueError("head_fusion must be mean, max, or min")
        if discard_ratio > 0:
            flat = fused.reshape(-1)
            count = int(flat.size * discard_ratio)
            if count:
                flat[np.argpartition(flat, count)[:count]] = 0
        identity = np.eye(fused.shape[0], dtype=np.float32)
        fused = (fused + identity) / 2.0
        fused = fused / np.clip(fused.sum(axis=-1, keepdims=True), 1e-12, None)
        matrices.append(fused)

    rollout = matrices[0]
    for matrix in matrices[1:]:
        rollout = matrix @ rollout
    class_attention = rollout[0, 1:]
    side = int(round(np.sqrt(class_attention.size)))
    if side * side != class_attention.size:
        raise ValueError("Patch token count is not a square; cannot form a 2D map.")
    heatmap = class_attention.reshape(side, side)
    heatmap -= heatmap.min()
    heatmap /= max(float(heatmap.max()), 1e-12)
    return heatmap


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB")
    scaled = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").resize(base.size, Image.Resampling.BICUBIC)
    color = Image.merge("RGB", (scaled, Image.new("L", base.size, 0), Image.new("L", base.size, 0)))
    return Image.blend(base, color, alpha)


def generate_attention_visualization(model, processor, image: Image.Image, output_path: str | Path) -> Path:
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    if not getattr(outputs, "attentions", None):
        raise RuntimeError("The checkpoint did not return attention tensors.")
    heatmap = attention_rollout(outputs.attentions)
    overlay = overlay_heatmap(image, heatmap)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output)
    return output
