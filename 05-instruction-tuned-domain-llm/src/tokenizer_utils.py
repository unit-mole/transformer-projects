"""Tokenization helpers for seq2seq instruction tuning."""
from __future__ import annotations

from typing import Any, Dict

from .prompt_templates import build_training_prompt


def tokenize_batch(
    examples: Dict[str, list[Any]],
    tokenizer: Any,
    *,
    max_input_length: int = 384,
    max_target_length: int = 192,
) -> Dict[str, Any]:
    prompts = [
        build_training_prompt(instruction, input_text, category)
        for instruction, input_text, category in zip(
            examples["instruction"], examples.get("input", [""] * len(examples["instruction"])), examples.get("category", [""] * len(examples["instruction"]))
        )
    ]
    model_inputs = tokenizer(prompts, max_length=max_input_length, truncation=True)
    labels = tokenizer(text_target=examples["output"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
