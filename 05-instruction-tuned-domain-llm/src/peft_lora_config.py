"""LoRA/PEFT configuration for FLAN-T5 sequence-to-sequence tuning."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from .config import LoraTrainingConfig


def build_lora_config(config: LoraTrainingConfig | None = None) -> Any:
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError("Install peft to build the LoRA configuration.") from exc

    cfg = config or LoraTrainingConfig()
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
    )


def lora_config_dict(config: LoraTrainingConfig | None = None) -> Dict[str, object]:
    return asdict(config or LoraTrainingConfig())
