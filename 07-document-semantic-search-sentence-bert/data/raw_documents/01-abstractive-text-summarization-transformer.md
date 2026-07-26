---
project_name: Abstractive Text Summarization Transformer
project_category: Transformer / NLP
document_type: project_readme
tags: summarization, bart, t5, rouge, bertscore, hugging-face-spaces
url: https://github.com/unit-mole/transformer-projects/tree/main/01-abstractive-text-summarization-transformer
---
# Abstractive Text Summarization Transformer

## Objective
Build an end-to-end abstractive summarization system that converts long articles into concise, fluent summaries. The project compares Transformer checkpoints, preserves factual meaning, and documents responsible use for generated text.

## Model and evaluation
The workflow uses BART or T5-style sequence-to-sequence Transformers. Evaluation includes ROUGE, BERTScore, compression ratio, latency, and qualitative error analysis for hallucination, repetition, and omitted facts.

## Deployment
The interactive demonstration is designed for Hugging Face Spaces. The repository includes a model card, dataset card, reproducible inference code, and sample summaries.
