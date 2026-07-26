from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, received {value!r}.") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, received {value!r}.") from exc


@dataclass(frozen=True)
class InferenceConfig:
    """Runtime settings for local use and Hugging Face Spaces."""

    model_id: str = os.getenv(
        "LONGDOCQA_MODEL_ID",
        "valhalla/longformer-base-4096-finetuned-squadv1",
    )
    max_length: int = _env_int("LONGDOCQA_MAX_LENGTH", 2048)
    stride: int = _env_int("LONGDOCQA_STRIDE", 256)
    max_answer_tokens: int = _env_int("LONGDOCQA_MAX_ANSWER_TOKENS", 48)
    minimum_confidence_proxy: float = _env_float(
        "LONGDOCQA_MIN_CONFIDENCE_PROXY", 0.002
    )
    max_document_characters: int = _env_int(
        "LONGDOCQA_MAX_DOCUMENT_CHARACTERS", 2_000_000
    )
    max_upload_mb: int = _env_int("LONGDOCQA_MAX_UPLOAD_MB", 10)
    device: str = os.getenv("LONGDOCQA_DEVICE", "auto")
    sample_directory: Path = PROJECT_ROOT / "data" / "sample_documents"

    def validate(self) -> "InferenceConfig":
        if not 512 <= self.max_length <= 4096:
            raise ValueError("max_length must be between 512 and 4096.")
        if not 0 <= self.stride < self.max_length:
            raise ValueError("stride must be non-negative and smaller than max_length.")
        if not 1 <= self.max_answer_tokens <= 256:
            raise ValueError("max_answer_tokens must be between 1 and 256.")
        if self.max_upload_mb <= 0:
            raise ValueError("max_upload_mb must be positive.")
        return self

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["sample_directory"] = str(data["sample_directory"])
        return data
