---
title: DocRank360 Two Stage Transformer Ranking
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
models:
  - Xenova/all-MiniLM-L6-v2
  - Xenova/ms-marco-MiniLM-L-6-v2
short_description: Browser-based bi-encoder retrieval and cross-encoder reranking.
tags:
  - transformers-js
  - semantic-search
  - information-retrieval
  - bi-encoder
  - cross-encoder
  - reranking
---

# DocRank360 — Two-Stage Transformer Search

**Real browser-based Transformer inference:** This application runs MiniLM
bi-encoder retrieval and MS MARCO cross-encoder reranking directly in the
visitor's browser using Transformers.js and ONNX Runtime Web. No server-side
inference API or paid Hugging Face compute is used.

## Browser models

- Bi-encoder: `Xenova/all-MiniLM-L6-v2`
- Cross-encoder: `Xenova/ms-marco-MiniLM-L-6-v2`
- Quantization: q8
- Hosting: Hugging Face Static Space

## What the demo shows

- semantic candidate retrieval;
- cross-encoder reranking;
- cosine and relevance scores;
- original rank, final rank, and rank movement;
- Recall@K, MRR@10, and nDCG@10 for labelled samples;
- browser model-loading and inference latency;
- downloadable JSON ranking results;
- limitations and responsible-use guidance.

## Portfolio links

- GitHub project:
  `https://github.com/unit-mole/transformer-projects/tree/main/03-cross-encoder-bi-encoder-ranking-system`
- Python implementation:
  `https://github.com/unit-mole/transformer-projects/tree/main/03-cross-encoder-bi-encoder-ranking-system/src`
- Evaluation notebook:
  `https://github.com/unit-mole/transformer-projects/blob/main/03-cross-encoder-bi-encoder-ranking-system/notebooks/retrieval_reranking_evaluation.ipynb`
- Model card:
  `https://github.com/unit-mole/transformer-projects/blob/main/03-cross-encoder-bi-encoder-ranking-system/MODEL_CARD.md`

## Responsible use

This educational demo can return incomplete, biased, irrelevant, or misleading
rankings. Do not enter private or confidential data. Do not use search or job
matching scores as the sole basis for hiring, rejection, compensation,
promotion, immigration, legal, or employment decisions.
