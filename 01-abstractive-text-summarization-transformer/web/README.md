---
title: Abstractive Text Summarization Transformer
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: static
app_build_command: npm run build
app_file: dist/index.html
pinned: false
license: mit
models:
  - Xenova/distilbart-cnn-12-6
short_description: Real browser-based DistilBART summarization with decoding controls, chunking, and latency metrics.
---

# Abstractive Text Summarization Transformer

A free Hugging Face **Static Space** that performs real Transformer inference inside the visitor's browser using **Transformers.js**, **ONNX Runtime Web**, and the browser-compatible `Xenova/distilbart-cnn-12-6` model.

## What makes this a real ML demo

- No mock summaries and no server-side inference API.
- The encoder-decoder Transformer is downloaded and executed in the browser.
- WebGPU is used when selected and available; WASM provides a broad compatibility fallback.
- Quantized ONNX weights reduce download and memory requirements.
- Generation controls expose beam search, output length, length penalty, and repetition control.
- Long inputs are split into token-aware chunks and optionally summarized in a second pass.

## How to use

1. Paste an English article or choose a bundled example.
2. Select the browser runtime.
3. Adjust summary-length and beam-search controls.
4. Select **Generate summary**.
5. Review latency, compression, token counts, chunk count, runtime, and model details.
6. Copy or download the generated summary.

## Models

| Layer | Model |
|---|---|
| Python implementation | `sshleifer/distilbart-cnn-12-6` |
| Static browser implementation | `Xenova/distilbart-cnn-12-6` |

The browser model is an ONNX/Transformers.js conversion of the same DistilBART checkpoint used by the Python project. It is not presented as a model trained by the portfolio author.

## Evaluation

The GitHub repository contains Python evaluation scripts for ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, compression ratio, and inference time. Metrics are shown only after an actual evaluation run; this Space does not fabricate scores.

## Transformer vs LSTM

The repository includes a strict comparison framework against the earlier LSTM Seq2Seq summarization project. Actual LSTM predictions are required before publishing comparison metrics.

## Responsible use

Generated summaries can omit context, distort details, or hallucinate. Do not paste private, confidential, sensitive, copyrighted, or personally identifiable text into a public Space. Do not use outputs as the sole basis for legal, medical, financial, safety-critical, academic, journalistic, or official decisions. Human review is required.

## Portfolio links

- GitHub repository: `https://github.com/unit-mole/transformer-projects`
- Project folder: `https://github.com/unit-mole/transformer-projects/tree/main/01-abstractive-text-summarization-transformer`
- Base Python model: `https://huggingface.co/sshleifer/distilbart-cnn-12-6`
- Browser ONNX model: `https://huggingface.co/Xenova/distilbart-cnn-12-6`
