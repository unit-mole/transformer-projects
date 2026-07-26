"""LoRA configuration for FLAN-T5 sequence-to-sequence adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoraSettings:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = field(default_factory=lambda: ("q", "v"))


def create_lora_config(settings: LoraSettings | None = None):
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise RuntimeError("Install the 'peft' package to configure LoRA.") from exc
    cfg = settings or LoraSettings()
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
    )
