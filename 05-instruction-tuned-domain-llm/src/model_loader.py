"""Lazy model and adapter loading for local use and Hugging Face Spaces."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import ModelConfig


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    base_model_id: str
    adapter_source: str
    merged: bool
    device: str


def _resolve_dtype(torch_module: Any, requested: str) -> Any:
    if requested == "auto":
        return "auto"
    mapping = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    if requested not in mapping:
        raise ValueError(f"Unsupported TORCH_DTYPE: {requested}")
    return mapping[requested]


def _adapter_source(config: ModelConfig) -> str:
    if config.adapter_id.strip():
        return config.adapter_id.strip()
    local_path = Path(config.local_adapter_path)
    if (local_path / "adapter_config.json").exists():
        return str(local_path)
    return ""


def load_model_and_tokenizer(
    config: Optional[ModelConfig] = None,
    *,
    merge_adapter: bool | None = None,
) -> LoadedModel:
    cfg = config or ModelConfig()
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install torch and transformers before loading the model.") from exc

    dtype = _resolve_dtype(torch, cfg.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model_id, **model_kwargs)

    adapter_source = _adapter_source(cfg)
    merged = False
    if adapter_source:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Install peft to load the LoRA adapter.") from exc
        model = PeftModel.from_pretrained(model, adapter_source, is_trainable=False)
        should_merge = merge_adapter if merge_adapter is not None else os.getenv("MERGE_ADAPTER", "false").lower() == "true"
        if should_merge:
            model = model.merge_and_unload()
            merged = True

    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device
    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        base_model_id=cfg.base_model_id,
        adapter_source=adapter_source or "none_base_model_fallback",
        merged=merged,
        device=device,
    )
