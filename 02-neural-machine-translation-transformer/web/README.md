---
title: English Hindi Neural Machine Translation
emoji: 🌐
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
fullWidth: true
short_description: English-Hindi Transformer translation and evaluation
---

# English–Hindi Neural Machine Translation — Static Space

This folder is a **free Hugging Face Static Space** deployment for Project 02.
It performs real MarianMT encoder-decoder inference inside the visitor's browser
using Transformers.js and ONNX Runtime. No Python server, paid Gradio compute,
or external inference API is required.

## Models

- English → Hindi: `Xenova/opus-mt-en-hi`
- Hindi → English: `Xenova/opus-mt-hi-en`
- Base models: `Helsinki-NLP/opus-mt-en-hi` and `Helsinki-NLP/opus-mt-hi-en`

The app lazily downloads only the selected directional model. The first request
can be slow because model files must be downloaded and cached in the browser.

## Deploy

Create a new Hugging Face Space, choose **Static → Blank** or
**Static → Transformers.js**, and upload the contents of this `web/` folder to
the root of the Space repository.

For the complete Python implementation, evaluation notebooks, tests, SacreBLEU,
chrF, latency analysis, and error analysis, use the parent GitHub project folder.
