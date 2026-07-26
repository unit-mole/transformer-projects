---
title: English Hindi Neural Machine Translation
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.20.0
python_version: 3.11
app_file: app.py
pinned: false
license: mit
suggested_hardware: cpu-basic
---

# English–Hindi Neural Machine Translation

Interactive MarianMT demo for English→Hindi and Hindi→English translation.

## How to use

1. Open **Sentence Translation**.
2. Paste English or Hindi text.
3. Keep direction on **Automatic** or select a manual direction.
4. Review the translation, detected language, confidence proxy, and latency.
5. Use **Batch Translation** for a CSV containing a text column.

## Models

- `Helsinki-NLP/opus-mt-en-hi`
- `Helsinki-NLP/opus-mt-hi-en`

Models are loaded lazily on the first request and are not trained during Space startup.

## Important limitation

The confidence value is a model-based proxy, not a guarantee of translation correctness. Human review is required. Do not submit sensitive or decision-critical text.

## Repository

`<YOUR_GITHUB_REPOSITORY_URL>`
