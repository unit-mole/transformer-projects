---
title: Abstractive Text Summarization Transformer
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: mit
models:
  - Xenova/distilbart-cnn-12-6
short_description: Browser-based DistilBART text summarization
---

# Abstractive Text Summarization Transformer

A browser-based DistilBART summarization demo using Transformers.js and ONNX
Runtime Web.

The GitHub workflow builds the Vite application before deployment. Hugging Face
receives prebuilt files and directly serves `index.html`, so no
`app_build_command` is used.

The live browser application uses `Xenova/distilbart-cnn-12-6`. The repository
also contains the Python fine-tuning and benchmark pipeline.
