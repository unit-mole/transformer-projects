---
project_name: Document Semantic Search with Sentence-BERT
project_category: Transformer / Information Retrieval
document_type: project_readme
tags: sentence-bert, semantic-search, cosine-similarity, github-pages, transformersjs, recall-at-k, mrr
url: https://github.com/unit-mole/transformer-projects/tree/main/07-document-semantic-search-sentence-bert
---
# Document Semantic Search with Sentence-BERT

## Objective
Create an entirely browser-based semantic search engine for public portfolio documentation. Sentence-BERT maps natural-language queries and document chunks into a shared 384-dimensional embedding space.

## Retrieval
The system preserves Markdown section metadata, generates normalized embeddings, applies optional category and document-type filters, calculates cosine similarity, and returns ranked cards with provenance. The interface reports query embedding latency, ranking latency, total latency, and the active retrieval mode.

## Deployment
The app is static HTML, CSS, and JavaScript deployed to GitHub Pages. Transformers.js runs a browser-compatible all-MiniLM-L6-v2 model. No Python backend, vector database, paid API, Streamlit, or Gradio service is required.

## Evaluation
Recall@K and MRR measure retrieval quality. Cosine-score analysis and manual relevance review identify high-similarity false positives, weak matches, and chunking problems.
