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
  - onnx-community/text_summarization-ONNX
short_description: Browser-based Transformer text summarization
---

# Abstractive Text Summarization Transformer

This Static Space performs real encoder-decoder Transformer inference directly
inside the visitor's browser.

## Browser deployment model

The live demo uses:

```text
onnx-community/text_summarization-ONNX
```

This is a fine-tuned T5-small summarization checkpoint with Transformers.js and
ONNX support. The application deliberately loads the full-precision FP32 model
through ONNX Runtime Web WASM. This avoids the incompatible `MatMulNBits`
quantized decoder graphs that repeatedly failed in the earlier DistilBART
browser export.

## Python model-development layer

The GitHub repository separately contains the original Project 01 DistilBART
work:

- pretrained DistilBART inference;
- RTX fine-tuning;
- 5,000 training, 500 validation, and 500 held-out test examples;
- ROUGE and BERTScore;
- Lead-3 and TextRank comparisons;
- latency, compression, and error analysis;
- generated JSON, CSV, Markdown, and PNG evidence.

The live T5 browser checkpoint and Python DistilBART benchmark are clearly
separated. Metrics produced by the Python notebook must not be represented as
T5 browser-model metrics.

## Runtime

```text
Backend: ONNX Runtime Web WASM
Precision: FP32
Server-side inference: none
Paid Hugging Face compute: none
```

The first load downloads several hundred megabytes of model files and stores
them in the browser cache.

Source repository:
https://github.com/unit-mole/transformer-projects/tree/main/01-abstractive-text-summarization-transformer
