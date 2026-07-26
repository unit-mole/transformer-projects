"""Lazy CPU/GPU-safe loading of the base model and optional LoRA adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLACEHOLDER_PREFIXES = ("YOUR_", "<", "REPLACE_")


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    base_model_id: str
    adapter_model_id: str | None
    mode: str
    device: str


def _is_real_adapter_id(value: str | None) -> bool:
    if not value:
        return False
    return not value.strip().startswith(PLACEHOLDER_PREFIXES)


def load_model(
    base_model_id: str | None = None,
    adapter_model_id: str | None = None,
    force_base: bool = False,
) -> LoadedModel:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers before loading the model.") from exc

    base_model_id = base_model_id or os.getenv("BASE_MODEL_ID", "google/flan-t5-small")
    adapter_model_id = adapter_model_id if adapter_model_id is not None else os.getenv("ADAPTER_MODEL_ID", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=dtype)
    base_model.to(device)

    mode = "base_model"
    effective_adapter: str | None = None
    if not force_base and _is_real_adapter_id(adapter_model_id):
        try:
            from peft import PeftModel
            base_model = PeftModel.from_pretrained(base_model, adapter_model_id)
            effective_adapter = adapter_model_id
            mode = "lora_adapter"
        except Exception as exc:
            raise RuntimeError(
                f"The LoRA adapter could not be loaded from '{adapter_model_id}'. "
                "Confirm that it is a valid PEFT repository compatible with the base model."
            ) from exc
    base_model.eval()
    return LoadedModel(
        model=base_model,
        tokenizer=tokenizer,
        base_model_id=base_model_id,
        adapter_model_id=effective_adapter,
        mode=mode,
        device=device,
    )
