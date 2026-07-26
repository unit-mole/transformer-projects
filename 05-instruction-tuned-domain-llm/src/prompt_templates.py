"""Prompt templates and sample prompts shared by training and inference."""

from __future__ import annotations

from typing import Final

SYSTEM_SCOPE: Final[str] = (
    "You are an educational ML and Data Science learning assistant. "
    "Answer clearly, stay within ML/DS topics, state uncertainty, and do not provide "
    "legal, medical, financial, immigration, or safety-critical advice."
)

PROMPT_CATEGORIES: Final[dict[str, list[str]]] = {
    "Concept explanation": [
        "Explain random forest in simple terms.",
        "What is overfitting and how can I reduce it?",
    ],
    "Algorithm comparison": [
        "Compare logistic regression and decision tree.",
        "Explain the difference between CNN and Transformer.",
    ],
    "Metric explanation": [
        "Explain precision vs recall with a quality analytics example.",
        "When should I use MAE instead of RMSE?",
    ],
    "Example generation": [
        "Generate a small Python example of a stratified train-test split.",
        "Show a simple scikit-learn pipeline with standardization.",
    ],
    "Interview answer": [
        "Give an interview-style answer: How do you detect overfitting?",
    ],
    "Workflow guidance": [
        "Explain a practical workflow for an imbalanced classification project.",
    ],
    "Quality analytics": [
        "Explain how machine learning could support quality case prioritization.",
    ],
}


def format_prompt(instruction: str, input_text: str = "", include_scope: bool = True) -> str:
    instruction = " ".join(str(instruction).split())
    input_text = " ".join(str(input_text).split())
    parts = []
    if include_scope:
        parts.append(f"Context: {SYSTEM_SCOPE}")
    parts.append(f"Instruction: {instruction}")
    if input_text:
        parts.append(f"Input: {input_text}")
    parts.append("Response:")
    return "\n\n".join(parts)


def all_examples() -> list[list[str]]:
    return [[prompt] for prompts in PROMPT_CATEGORIES.values() for prompt in prompts]
