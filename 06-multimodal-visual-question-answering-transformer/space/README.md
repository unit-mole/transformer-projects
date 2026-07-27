---
title: Multimodal Visual Question Answering Transformer
emoji: 🖼️
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Browser-based image question answering with Transformers.js and WebGPU.
models:
  - HuggingFaceTB/SmolVLM-256M-Instruct
---

# Multimodal Visual Question Answering Transformer

Upload a safe, non-sensitive image and ask a natural-language question about
it. The model runs in the browser through Transformers.js and WebGPU; the
Static Space does not run a Python server.

The first model load is large and can take several minutes. A modern
WebGPU-capable Chromium browser and a stable connection are recommended. The
app uses the same SmolVLM-256M-Instruct model class, processor flow, chat
template, and stable WebGPU fp32 configuration used in Hugging Face’s official
SmolVLM browser example. Failed runs show troubleshooting details and can be retried.

**Responsible use:** This educational demo can return incorrect, biased, or
misleading answers. Do not upload private, confidential, medical, identity,
workplace, or other sensitive images. Do not use it for surveillance, identity
recognition, or high-stakes decisions.
