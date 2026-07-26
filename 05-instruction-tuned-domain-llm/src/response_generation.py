"""Text generation utilities with explicit, reproducible settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 160
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.1


def generate_response(loaded, prompt: str, settings: GenerationSettings | None = None) -> tuple[str, dict[str, Any]]:
    import torch

    cfg = settings or GenerationSettings()
    tokenizer = loaded.tokenizer
    model = loaded.model
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
    encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
    do_sample = cfg.temperature > 0
    generation_kwargs = {
        "max_new_tokens": int(cfg.max_new_tokens),
        "do_sample": do_sample,
        "repetition_penalty": float(cfg.repetition_penalty),
        "num_beams": 1,
    }
    if do_sample:
        generation_kwargs.update(
            temperature=max(float(cfg.temperature), 1e-5),
            top_p=float(cfg.top_p),
        )
    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return text, {
        "model_mode": loaded.mode,
        "base_model": loaded.base_model_id,
        "adapter": loaded.adapter_model_id,
        "device": loaded.device,
        "generation": cfg.__dict__,
    }
