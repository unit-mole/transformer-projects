"""Prompt templates shared by training and inference."""
from __future__ import annotations

from typing import Dict

SYSTEM_SCOPE = (
    "You are an educational ML and Data Science learning assistant. "
    "Answer clearly, stay within ML/DS, state uncertainty, and do not provide "
    "legal, medical, financial, immigration, or safety-critical advice."
)

CATEGORY_GUIDANCE: Dict[str, str] = {
    "Concept explanation": "Explain the concept in plain language, then give one practical example.",
    "Algorithm comparison": "Compare the methods using strengths, limitations, and when to use each.",
    "Metric explanation": "Define the metric, explain its interpretation, and mention one limitation.",
    "Example generation": "Provide a small, concrete example and explain what it demonstrates.",
    "Beginner-friendly explanation": "Use simple language and avoid unnecessary jargon.",
    "Interview-style answer": "Give a concise, structured answer suitable for an interview.",
    "Small code example": "Provide a minimal Python example and explain the key lines.",
    "Data Science workflow": "Describe the steps in order and include validation considerations.",
    "ML project guidance": "Provide practical project guidance with clear next steps.",
    "Quality analytics": "Use a non-confidential quality analytics example and explain the modeling choice.",
}


def clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def build_training_prompt(instruction: str, input_text: str = "", category: str = "") -> str:
    instruction = clean_text(instruction)
    input_text = clean_text(input_text)
    category = clean_text(category)
    parts = [f"System: {SYSTEM_SCOPE}"]
    if category:
        guidance = CATEGORY_GUIDANCE.get(category, "Answer the instruction accurately and clearly.")
        parts.append(f"Category: {category}\nGuidance: {guidance}")
    parts.append(f"Instruction: {instruction}")
    if input_text:
        parts.append(f"Input: {input_text}")
    parts.append("Response:")
    return "\n\n".join(parts)


def build_inference_prompt(instruction: str, input_text: str = "", category: str = "") -> str:
    return build_training_prompt(instruction, input_text, category)
