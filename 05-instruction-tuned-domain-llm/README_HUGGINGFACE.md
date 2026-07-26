---
title: ML Data Science Instruction Tuned Assistant
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.13.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
suggested_hardware: cpu-basic
---

# ML & Data Science Instruction-Tuned Assistant

A portfolio demonstration of FLAN-T5 sequence-to-sequence instruction tuning with LoRA / PEFT, custom instruction data, transparent evaluation, and a Gradio interface.

## Use the demo

Choose a prompt category, enter an ML or Data Science question, optionally add context, adjust generation settings, and generate a response. The adapter repository must be configured through `ADAPTER_MODEL_ID`; otherwise the app transparently uses the base model.

## Responsible use

This educational demo may generate incomplete, incorrect, outdated, biased, or hallucinated responses. It is not for legal, medical, financial, immigration, safety-critical, official, or autonomous decisions. Do not enter private, confidential, proprietary, copyrighted, or personally identifiable information.

## Links

- GitHub: `https://github.com/YOUR_GITHUB_USERNAME/transformer-projects/tree/main/05-instruction-tuned-domain-llm`
- Adapter: `https://huggingface.co/YOUR_HF_USERNAME/ml-ds-instruction-tuned-flan-t5-small-lora`
