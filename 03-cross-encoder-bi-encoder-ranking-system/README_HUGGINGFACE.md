---
title: DocRank360 Two Stage Search Ranking
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
models:
  - sentence-transformers/all-MiniLM-L6-v2
  - cross-encoder/ms-marco-MiniLM-L-6-v2
preload_from_hub:
  - sentence-transformers/all-MiniLM-L6-v2
  - cross-encoder/ms-marco-MiniLM-L-6-v2
suggested_hardware: cpu-basic
---

# DocRank360 — Bi-Encoder Retrieval + Cross-Encoder Reranking

Enter a query, retrieve top candidates with MiniLM sentence embeddings, and
rerank them with an MS MARCO MiniLM cross-encoder.

## How to use

1. Enter a query or choose a sample.
2. Select candidate K.
3. Select rerank K.
4. Choose bi-encoder-only or two-stage mode.
5. Compare scores, rank movement, and latency.

## Models

- Bi-encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Vector search: normalized NumPy cosine similarity

## Output

The demo shows candidate results, reranked results, bi-encoder scores,
cross-encoder scores, rank movement, and measured stage latency.

## Evaluation

Run the repository evaluation script to calculate Recall@K, MRR@10, nDCG@10,
reranking improvement, and latency. The interface does not display invented
metrics.

## Responsible use

This educational demo can produce biased, incomplete, irrelevant, or misleading
rankings. Do not upload private or confidential content. Do not use job-ranking
scores as the sole basis for employment, immigration, legal, compensation, or
hiring decisions.

## Limitations

The committed dataset is a small public-safe synthetic sample. Cross-encoder
scores are not probabilities, and results do not establish production
performance or fairness.

GitHub: `https://github.com/<YOUR_GITHUB_USERNAME>/transformer-projects`
