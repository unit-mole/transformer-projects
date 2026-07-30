---
title: ML Data Science Instruction Tuned Assistant
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
---

# ML & Data Science Instruction-Tuned Assistant

A Gradio portfolio demo using `google/flan-t5-base` with a reviewed LoRA/PEFT adapter trained on a custom ML and Data Science instruction dataset.

## Responsible Use

Educational demonstration only. The model may generate incomplete, incorrect, outdated, biased, or hallucinated content. Do not use it for legal, medical, financial, immigration, safety-critical, or official decisions. Do not enter confidential, proprietary, copyrighted, or personally identifiable information. Review every output.

## How to Use

1. Select a prompt category.
2. Enter an ML or Data Science question.
3. Add optional context or format constraints.
4. Adjust generation settings when needed.
5. Generate the response.
6. Confirm the metadata reports `lora_adapter`.
7. Review the answer before using it.

## Model Details

- Base model: `google/flan-t5-base`
- Architecture: encoder-decoder Transformer
- Fine-tuning: LoRA through Hugging Face PEFT
- Task type: sequence-to-sequence language modeling
- Adapter: set with the `ADAPTER_ID` Space variable or store locally in `models/lora_adapter/`
- Prompt format: system scope + category guidance + instruction + optional input + response marker

## Training Data

The project begins with 93 self-authored seed examples and expands toward approximately 600 reviewed ML/Data Science instructions. The training corpus is validated, de-duplicated, checked against an independent benchmark, stratified into splits, and sampled for human review.

## Evaluation

The selected adapter is compared with base FLAN-T5 on the same 80 held-out prompts. Published results can include:

- instruction adherence;
- response-quality rubric;
- BERTScore F1;
- ROUGE-L F1;
- sentence-embedding semantic similarity;
- latency;
- hallucination-risk flags;
- category-level metrics;
- base-versus-LoRA deltas, win rates, and bootstrap confidence intervals;
- human factuality and preference review.

Only results produced by the deployed adapter should be shown. Similarity metrics and heuristic flags do not prove factual correctness.

## Limitations

FLAN-T5-base remains a compact model with limited reasoning and knowledge. The custom data can contain style bias or errors despite review. CPU Spaces may have slower cold-start and generation latency. This is not a production expert or factual authority.

- GitHub: `https://github.com/<your-github-username>/transformer-models-projects`
- Adapter: `https://huggingface.co/<your-huggingface-username>/flan-t5-base-ml-ds-lora`
