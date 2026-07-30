"""Lazy end-to-end inference pipeline used by Gradio and scripts."""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from .config import ModelConfig
from .model_loader import LoadedModel, load_model_and_tokenizer
from .response_generation import generate_response

OUT_OF_SCOPE_TERMS = {
    "diagnose", "prescription", "lawsuit", "visa advice", "investment advice", "tax advice",
    "medical treatment", "immigration strategy", "legal advice",
}


class InstructionAssistant:
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig()
        self._loaded: Optional[LoadedModel] = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._loaded is not None

    def load(self) -> LoadedModel:
        if self._loaded is None:
            with self._lock:
                if self._loaded is None:
                    self._loaded = load_model_and_tokenizer(self.config)
        return self._loaded

    @staticmethod
    def scope_message(instruction: str) -> str:
        lower = instruction.lower()
        if any(term in lower for term in OUT_OF_SCOPE_TERMS):
            return (
                "This educational demo is limited to Machine Learning and Data Science. "
                "It does not provide legal, medical, financial, immigration, or safety-critical advice."
            )
        return ""

    def generate(
        self,
        instruction: str,
        category: str = "Concept explanation",
        input_text: str = "",
        max_new_tokens: int = 160,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> Dict[str, object]:
        instruction = str(instruction or "").strip()
        if not instruction:
            return {"response": "Please enter an ML or Data Science question.", "latency_seconds": 0.0, "model_mode": "not_loaded"}
        scope = self.scope_message(instruction)
        if scope:
            return {"response": scope, "latency_seconds": 0.0, "model_mode": "scope_guard"}

        loaded = self.load()
        start = time.perf_counter()
        response = generate_response(
            loaded.model,
            loaded.tokenizer,
            instruction,
            input_text,
            category,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_input_length=self.config.max_input_length,
        )
        latency = time.perf_counter() - start
        return {
            "response": response,
            "latency_seconds": round(latency, 4),
            "model_mode": "lora_adapter" if loaded.adapter_source != "none_base_model_fallback" else "base_model_fallback",
            "base_model": loaded.base_model_id,
            "adapter": loaded.adapter_source,
            "device": loaded.device,
            "merged": loaded.merged,
        }
