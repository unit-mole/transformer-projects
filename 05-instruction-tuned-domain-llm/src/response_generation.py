"""Response generation helpers shared by the app and evaluation pipeline."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from .config import GenerationConfig, ModelConfig
from .prompt_templates import build_inference_prompt


def generation_kwargs(
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> Dict[str, Any]:
    do_sample = temperature > 0.05
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
        "do_sample": do_sample,
        "no_repeat_ngram_size": 3,
        "length_penalty": 1.0,
        "early_stopping": True,
    }
    if do_sample:
        kwargs.update({"temperature": float(temperature), "top_p": float(top_p), "num_beams": 1})
    else:
        kwargs.update({"num_beams": 4})
    return kwargs


def generate_response(
    model: Any,
    tokenizer: Any,
    instruction: str,
    input_text: str = "",
    category: str = "",
    *,
    max_new_tokens: int = 220,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.12,
    max_input_length: int | None = None,
) -> str:
    if not str(instruction).strip():
        raise ValueError("Instruction cannot be empty.")

    prompt = build_inference_prompt(instruction, input_text, category)
    limit = max_input_length or ModelConfig().max_input_length
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=limit)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    kwargs = generation_kwargs(max_new_tokens, temperature, top_p, repetition_penalty)
    with __import__("torch").inference_mode():
        output_ids = model.generate(**encoded, **kwargs)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return response or "I could not generate a useful answer. Please rephrase the ML or Data Science question."


def default_generation_config() -> Dict[str, object]:
    return asdict(GenerationConfig())
