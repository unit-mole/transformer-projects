from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class LoadedVilt:
    processor: Any
    model: Any
    device: str
    model_id: str

def resolve_device(torch_module: Any, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if getattr(torch_module.cuda, "is_available", lambda: False)():
        return "cuda"
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is not None and getattr(mps, "is_available", lambda: False)():
        return "mps"
    return "cpu"

def load_vilt_model(
    model_id: str = "dandelin/vilt-b32-finetuned-vqa",
    device: str = "auto",
) -> LoadedVilt:
    try:
        import torch
        from transformers import ViltForQuestionAnswering, ViltProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Install the full runtime dependencies with `pip install -r requirements.txt`."
        ) from exc

    resolved = resolve_device(torch, device)
    processor = ViltProcessor.from_pretrained(model_id)
    model = ViltForQuestionAnswering.from_pretrained(model_id)
    model.to(resolved)
    model.eval()
    return LoadedVilt(processor=processor, model=model, device=resolved, model_id=model_id)
