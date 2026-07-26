from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class VQAConfig:
    python_model_id: str = "dandelin/vilt-b32-finetuned-vqa"
    static_model_id: str = "Xenova/moondream2"
    max_question_chars: int = 300
    max_image_megapixels: float = 25.0
    max_new_tokens: int = 48
    device: str = "auto"

DEFAULT_CONFIG = VQAConfig()
