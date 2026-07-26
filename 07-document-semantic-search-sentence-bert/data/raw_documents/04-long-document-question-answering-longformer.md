---
project_name: Long-Document Question Answering with Longformer
project_category: Transformer / NLP
document_type: model_card
tags: longformer, long-document, question-answering, long-context, hugging-face-spaces
url: https://github.com/unit-mole/transformer-projects/tree/main/04-long-document-question-answering-longformer
---
# Long-Document Question Answering with Longformer

## Problem
Standard Transformer context windows can truncate long reports. This project uses a Longformer-style architecture and sliding-window preprocessing to answer questions from lengthy technical documents.

## Evaluation
Answer quality is assessed with exact match, token-level F1, unanswerable-question handling, context coverage, and latency. Error analysis distinguishes retrieval failure, span-selection failure, and ambiguous questions.

## Deployment
The public demo is hosted on Hugging Face Spaces and uses only public sample documents.
