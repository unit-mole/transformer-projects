---
title: Long Document QA Browser
emoji: 📄
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Browser-based long-document QA with grounded evidence
models:
  - Xenova/distilbert-base-cased-distilled-squad
datasets:
  - allenai/qasper
---

# Long-Document Question Answering — Project 04

This credit-free Static Space runs genuine extractive question-answering
inference in the visitor's browser using Transformers.js, ONNX Runtime, and a
browser-compatible DistilBERT QA checkpoint.

The associated Python project fine-tuned and evaluated Longformer on an
extractive QASPER subset. That model is available separately at:

https://huggingface.co/anmol-unitmole/longformer-qasper-document-qa

The complete source project and evaluation artifacts are available at:

https://github.com/unit-mole/transformer-projects/tree/main/04-long-document-question-answering-longformer

## Architecture disclosure

- **Live browser model:** `Xenova/distilbert-base-cased-distilled-squad`
- **Evaluated Python model:** `anmol-unitmole/longformer-qasper-document-qa`

The Static Space does not claim that Longformer runs in the browser. The browser
application uses retrieval and overlapping chunks to apply a compact QA
Transformer to long documents without a paid Python server.
