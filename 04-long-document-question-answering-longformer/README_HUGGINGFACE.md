---
title: Long Document QA Longformer
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.20.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
models:
  - valhalla/longformer-base-4096-finetuned-squadv1
---

# Long-Document Question Answering with Longformer

Upload a TXT, Markdown, CSV, or selectable-text PDF document and ask a focused
question. The application returns an extractive answer, an uncalibrated model
confidence proxy, the supporting paragraph, highlighted evidence, document
length, window count, and latency.

## Model

`valhalla/longformer-base-4096-finetuned-squadv1`

The project does not claim new fine-tuning. It adds tokenizer-aware overlapping
windows, span selection across windows, evidence mapping, evaluation, and a
Gradio interface.

## How to use

1. Select a safe sample or upload a non-sensitive document.
2. Enter a question whose answer is stated explicitly in the document.
3. Choose a runtime token window and overlap.
4. Click **Ask Question**.
5. Review the answer and highlighted supporting paragraph together.

## Output

- Answer
- Model confidence proxy
- Supporting paragraph
- Highlighted evidence
- Context and latency diagnostics

## Responsible use

This demo can return incorrect or unsupported answers. Confidence is not a
calibrated probability. Do not upload private, confidential, proprietary,
copyrighted, sensitive, or personally identifiable documents. Human review is
required before any real-world use.

## Links

- GitHub: `https://github.com/<YOUR_USERNAME>/transformer-models-projects`
- Model card: `MODEL_CARD.md`
- Portfolio: `https://github.com/<YOUR_USERNAME>`
