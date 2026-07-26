---
project_name: Cross-Encoder and Bi-Encoder Ranking System
project_category: Transformer / Information Retrieval
document_type: project_readme
tags: sentence-transformers, bi-encoder, cross-encoder, reranking, information-retrieval
url: https://github.com/unit-mole/transformer-projects/tree/main/03-cross-encoder-bi-encoder-ranking-system
---
# Cross-Encoder and Bi-Encoder Ranking System

## Objective
Build a two-stage search-ranking pipeline. A Sentence-BERT bi-encoder converts queries and documents into reusable embeddings for efficient candidate retrieval. A cross-encoder then jointly scores each query-document pair for more precise reranking.

## Architecture
The bi-encoder retrieves a broad top-K shortlist with cosine similarity. The cross-encoder reranks only those candidates, balancing retrieval speed and relevance quality. Evaluation includes Recall@K, MRR, NDCG, reranking lift, and latency by candidate-set size.

## Portfolio value
The project demonstrates production-oriented information retrieval, embedding indexes, semantic search, reranking, and evaluation beyond simple accuracy.
