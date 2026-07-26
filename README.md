# Transformer Projects

A professional Transformer and multimodal-AI portfolio covering summarization,
translation, ranking, long-document question answering, instruction-tuned
language models, visual question answering, semantic search, Vision
Transformers, CLIP retrieval, and retrieval-augmented generation.

> **Career positioning:** I am a Quality Data Scientist building an applied
> Machine Learning, NLP, Generative AI, Computer Vision, and Multimodal AI
> portfolio with reproducible code, evaluation, CI, and free deployment demos.

[![Repository](https://img.shields.io/badge/GitHub-transformer--projects-181717?logo=github)](https://github.com/unit-mole/transformer-projects)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Static%20Spaces-FFD21E?logo=huggingface)](https://huggingface.co/anmol-unitmole)

## Completed projects

| # | Project | Core capability | Deployment |
|---:|---|---|---|
| 01 | [Abstractive Text Summarization Transformer](01-abstractive-text-summarization-transformer/) | Transformer summarization and evaluation | Hugging Face Static Space |
| 02 | [Neural Machine Translation Transformer](02-neural-machine-translation-transformer/) | English–Hindi sequence-to-sequence translation | Hugging Face Static Space |
| 03 | [Cross-Encoder / Bi-Encoder Ranking System](03-cross-encoder-bi-encoder-ranking-system/) | Two-stage semantic retrieval and reranking | Hugging Face Static Space |
| 04 | [Long-Document Question Answering with Longformer](04-long-document-question-answering-longformer/) | Long-context extractive QA | Hugging Face Static Space |
| 05 | [Instruction-Tuned Domain LLM](05-instruction-tuned-domain-llm/) | Domain instruction following and responsible generation | Hugging Face Static Space |
| 06 | [Multimodal Visual Question Answering Transformer](06-multimodal-visual-question-answering-transformer/) | Image + natural-language question answering | Hugging Face Static Space |

## Roadmap

| # | Planned project |
|---:|---|
| 07 | Document Semantic Search with Sentence-BERT |
| 08 | Image Classification with Vision Transformer |
| 09 | Vision-Language Image–Text Retrieval with CLIP |
| 10 | AI Portfolio RAG Assistant |

## Repository organization

Each numbered project contains its own documentation, source modules,
notebooks, tests, data notes, model and dataset cards, evaluation artifacts, and
deployment files. Project-specific GitHub Actions workflows live in
`.github/workflows/`.

```text
transformer-projects/
├── .github/workflows/
├── 01-abstractive-text-summarization-transformer/
├── 02-neural-machine-translation-transformer/
├── 03-cross-encoder-bi-encoder-ranking-system/
├── 04-long-document-question-answering-longformer/
├── 05-instruction-tuned-domain-llm/
├── 06-multimodal-visual-question-answering-transformer/
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── SECURITY.md
```

## Technology stack

Python, PyTorch, Hugging Face Transformers, Transformers.js, ONNX Runtime Web,
multimodal vision-language models, ViLT, Moondream2, Pillow, pandas, NumPy,
pytest, HTML, CSS, JavaScript, WebGPU, Hugging Face Static Spaces, and GitHub
Actions.

## Setup

Clone the repository, open the numbered project you want to explore, and follow
its README. Large datasets, model checkpoints, and browser model weights are not
committed; they are loaded from documented public sources.

## Responsible use

These projects are educational portfolio demonstrations. Model output can be
incorrect, biased, incomplete, or misleading. Do not use the demos for medical,
legal, financial, surveillance, identity verification, employment, insurance,
security, or other high-stakes decisions. Never upload private or confidential
content to a public demo.
