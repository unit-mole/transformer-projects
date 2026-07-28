---
title: Multimodal Visual Question Answering Transformer
emoji: 🖼️
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Browser VQA with SmolVLM, confidence diagnostics, and a 60-pair evaluation lab.
models:
  - HuggingFaceTB/SmolVLM-256M-Instruct
tags:
  - transformers
  - multimodal-ai
  - visual-question-answering
  - webgpu
  - transformers-js
---

# Multimodal Visual Question Answering Transformer

Upload a safe, non-sensitive image and ask a natural-language question. The
model runs through Transformers.js and WebGPU inside the browser; this Static
Space does not run a Python server.

The interface reports a generated answer, question and answer types, latency,
and an optional generation confidence proxy calculated from token scores. The
proxy is not calibrated as a probability that the answer is factually correct.

The Evaluation Lab runs a balanced 60-pair synthetic VQA suite and reports
overall accuracy, category-wise accuracy, answer failure rate, latency
statistics, downloadable JSON results, and failure examples. It is a portfolio
diagnostic rather than an official VQA v2 benchmark.

**Responsible use:** The demo can return incorrect, biased, or misleading
answers. Do not upload private, confidential, medical, identity, workplace, or
other sensitive images. Do not use it for surveillance, identification, or
high-stakes decisions.
