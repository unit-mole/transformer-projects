"""Tokenization helpers for sequence-to-sequence instruction tuning."""

from __future__ import annotations


def tokenize_batch(batch, tokenizer, max_source_length: int = 384, max_target_length: int = 192):
    model_inputs = tokenizer(
        batch["prompt"],
        max_length=max_source_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=batch["target"],
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
