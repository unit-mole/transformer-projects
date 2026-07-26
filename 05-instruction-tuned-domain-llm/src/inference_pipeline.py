"""Application-facing inference pipeline with lazy model caching and scope checks."""

from __future__ import annotations

import os
import time
from functools import lru_cache

from .model_loader import load_model
from .prompt_templates import SYSTEM_SCOPE, format_prompt
from .response_generation import GenerationSettings, generate_response

OUT_OF_SCOPE_TERMS = {
    "medical diagnosis", "legal advice", "immigration case", "stock recommendation",
    "prescription", "lawsuit", "visa decision",
}


def validate_user_prompt(text: str) -> tuple[bool, str]:
    clean = " ".join(str(text).split())
    if not clean:
        return False, "Enter an ML or Data Science learning question."
    if len(clean) > 1200:
        return False, "Keep the prompt under 1,200 characters for this public demo."
    lower = clean.lower()
    if any(term in lower for term in OUT_OF_SCOPE_TERMS):
        return False, "This educational demo is limited to ML and Data Science topics and cannot provide high-stakes advice."
    return True, clean


@lru_cache(maxsize=2)
def _cached_model(force_base: bool):
    return load_model(force_base=force_base)


def run_inference(
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 160,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    force_base: bool = False,
):
    ok, message = validate_user_prompt(instruction)
    if not ok:
        return message, {"status": "blocked", "scope": SYSTEM_SCOPE}
    start = time.perf_counter()
    loaded = _cached_model(bool(force_base))
    response, metadata = generate_response(
        loaded,
        format_prompt(message, input_text),
        GenerationSettings(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
        ),
    )
    metadata.update({"status": "ok", "latency_seconds": round(time.perf_counter() - start, 3)})
    return response, metadata
