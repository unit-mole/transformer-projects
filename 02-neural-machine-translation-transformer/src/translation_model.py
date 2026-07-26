from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from .confidence_scoring import (
    ConfidenceProxy,
    heuristic_confidence,
    sequence_score_confidence,
)
from .config import load_model_metadata


@dataclass(frozen=True)
class ModelTranslation:
    translated_text: str
    confidence: ConfidenceProxy
    latency_seconds: float
    model_id: str
    input_tokens: int
    output_tokens: int
    device: str


class MarianTranslationEngine:
    """Lazy, cached wrapper around two directional MarianMT models."""

    VALID_DIRECTIONS = {"en_hi", "hi_en"}

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = metadata or load_model_metadata()
        self.generation = self.metadata["generation_config"]
        self._bundles: dict[str, tuple[Any, Any, Any, str]] = {}

    def _model_id(self, direction: str) -> str:
        if direction == "en_hi":
            return self.metadata["en_hi_model_id"]
        if direction == "hi_en":
            return self.metadata["hi_en_model_id"]
        raise ValueError(f"Unsupported direction: {direction}")

    @staticmethod
    def _dependencies() -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch and transformers are required for model inference. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        return torch, AutoModelForSeq2SeqLM, AutoTokenizer

    def _load(self, direction: str) -> tuple[Any, Any, Any, str]:
        if direction not in self.VALID_DIRECTIONS:
            raise ValueError(f"Unsupported direction: {direction}")
        if direction in self._bundles:
            return self._bundles[direction]

        torch, auto_model, auto_tokenizer = self._dependencies()
        model_id = self._model_id(direction)
        tokenizer = auto_tokenizer.from_pretrained(model_id)
        model = auto_model.from_pretrained(model_id)

        requested = os.getenv("NMT_DEVICE", "auto").lower()
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("NMT_DEVICE=cuda was requested, but CUDA is unavailable.")
        device = (
            requested
            if requested in {"cpu", "cuda", "mps"}
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.to(device)
        model.eval()
        self._bundles[direction] = (torch, tokenizer, model, device)
        return self._bundles[direction]

    def translate(self, text: str, direction: str) -> ModelTranslation:
        torch, tokenizer, model, device = self._load(direction)
        model_id = self._model_id(direction)

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.generation["max_source_length"]),
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        generation_kwargs = {
            "num_beams": int(self.generation["num_beams"]),
            "max_new_tokens": int(self.generation["max_new_tokens"]),
            "early_stopping": bool(self.generation["early_stopping"]),
            "length_penalty": float(self.generation["length_penalty"]),
            "no_repeat_ngram_size": int(self.generation["no_repeat_ngram_size"]),
            "renormalize_logits": bool(self.generation["renormalize_logits"]),
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        latency = time.perf_counter() - start

        sequence = generated.sequences[0]
        translated_text = tokenizer.decode(sequence, skip_special_tokens=True).strip()

        sequence_scores = getattr(generated, "sequences_scores", None)
        if sequence_scores is not None and len(sequence_scores):
            confidence = sequence_score_confidence(float(sequence_scores[0].item()))
        else:
            unk_id = getattr(tokenizer, "unk_token_id", None)
            unknown_count = (
                int((sequence == unk_id).sum().item()) if unk_id is not None else 0
            )
            confidence = heuristic_confidence(
                text,
                translated_text,
                unknown_token_count=unknown_count,
            )

        input_tokens = int(encoded["input_ids"].shape[-1])
        output_tokens = int(sequence.shape[-1])

        return ModelTranslation(
            translated_text=translated_text,
            confidence=confidence,
            latency_seconds=latency,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            device=device,
        )
