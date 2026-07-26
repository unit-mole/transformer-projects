# Project 03 Portfolio and Deployment Roadmap

## Static deployment does not reduce the value of Project 03

Static deployment changes **where inference runs**, not whether genuine
Transformer inference occurs.

- **Gradio:** Python model inference runs through a Python process on hosted
  compute.
- **Static + Transformers.js:** quantized ONNX Transformer models run directly
  inside the visitor's browser through ONNX Runtime Web.

Project 03 therefore remains a genuine Transformer and information-retrieval
portfolio project. The live demo performs real MiniLM embedding generation,
cosine candidate retrieval, and cross-encoder query-document scoring.

## Best portfolio approach

Use a three-part structure:

| Portfolio component | Purpose |
|---|---|
| GitHub repository | Full Python project, Gradio comparison app, evaluation, tests, notebooks, outputs, CI and engineering structure |
| Hugging Face Model repository | Pipeline card, base-model attribution, configuration, dataset, metrics, limitations and responsible-use documentation |
| Hugging Face Static Space | Free interactive browser demo using Transformers.js and ONNX Runtime Web |

This is stronger than showing only a Gradio application because it demonstrates
both Python ML engineering and browser-based Transformer deployment.

## 1. Keep the complete Python project

Do not remove:

```text
app.py
gradio_app.py
src/
scripts/
tests/
notebooks/
outputs/
requirements.txt
MODEL_CARD.md
Dockerfile
```

These files demonstrate:

- Sentence Transformers;
- MiniLM embeddings;
- cross-encoder pair scoring;
- two-stage information retrieval;
- preprocessing and Unicode handling;
- NumPy vector indexing;
- Recall@K;
- MRR@10;
- nDCG@10;
- reranking improvement;
- stage-specific latency;
- automated tests;
- manual error analysis;
- local Gradio application;
- Docker packaging.

The Python implementation remains the technical foundation.

## 2. Add a separate Static frontend

The deployment layer is intentionally isolated:

```text
03-cross-encoder-bi-encoder-ranking-system/
│
├── app.py
├── gradio_app.py
├── src/
├── scripts/
├── tests/
├── notebooks/
├── outputs/
├── model_hub/
│
└── web/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── public/
    │   ├── README.md
    │   └── data/
    └── src/
        ├── constants.js
        ├── data-loader.js
        ├── metrics.js
        ├── export-results.js
        ├── ranking-engine.js
        ├── ui.js
        ├── main.js
        └── styles.css
```

The Static Space runs:

```text
User query
    ↓
Xenova/all-MiniLM-L6-v2
    ↓
Normalized query and document embeddings
    ↓
Cosine candidate retrieval
    ↓
Xenova/ms-marco-MiniLM-L-6-v2
    ↓
Cross-encoder reranking
    ↓
Rank movement + quality metrics + latency
```

## What the Static demo displays

To make the Transformer work visible to recruiters, the demo includes:

- original Python base-model names;
- browser-compatible ONNX model names;
- two-stage architecture explanation;
- candidate-K and rerank-K controls;
- bi-encoder-only mode;
- two-stage mode;
- query input and sample queries;
- bi-encoder similarity scores;
- cross-encoder relevance scores;
- retrieval rank;
- final rank;
- rank movement;
- model-loading progress;
- browser setup latency;
- query-embedding latency;
- retrieval latency;
- reranking latency;
- total latency;
- Recall@K;
- MRR@10 before and after reranking;
- nDCG@10 before and after reranking;
- downloadable JSON results;
- limitations;
- responsible-use guidance.

This makes the demo look like an ML ranking system rather than a generic search
webpage.

## Recommended presentation statement

The Static Space should state:

> **Real browser-based Transformer inference:** This application runs MiniLM
> bi-encoder retrieval and MS MARCO cross-encoder reranking directly in the
> browser using Transformers.js and ONNX Runtime Web. No server-side inference
> API or paid Space runtime is used.

## Links displayed in the Static Space

```text
View Python implementation
View evaluation notebook
View model card
View ranking metrics and outputs
View GitHub repository
```

## 3. Hugging Face Model Hub strategy

### Current truthful repository

Create:

```text
anmol-unitmole/docrank360-ranking-pipeline-card
```

This repository documents the complete system without pretending that the base
weights were trained by the portfolio author.

Include:

- Python and browser model names;
- base-model attribution;
- two-stage architecture;
- dataset;
- preprocessing;
- training status;
- Recall@K, MRR@10 and nDCG@10;
- latency;
- ranking examples;
- limitations;
- responsible use;
- GitHub and Static Space links.

Ready files are under:

```text
model_hub/pipeline-card/
```

### Future repositories after real fine-tuning

Only create these after actual fine-tuning or your own validated model
conversion:

```text
anmol-unitmole/docrank360-bi-encoder-retrieval
anmol-unitmole/docrank360-cross-encoder-reranker
```

Do not imply that a pretrained or community-converted model was trained by you.

## Final portfolio setup

```text
GitHub
└── Complete Python ML and information-retrieval project

Hugging Face Model Hub
└── Pipeline card, configuration, evaluation and limitations

Hugging Face Static Space
└── Live Transformers.js two-stage ranking demo
```

## Skills demonstrated

- Transformer inference;
- Sentence-BERT;
- MiniLM;
- semantic search;
- dense retrieval;
- cross-encoder reranking;
- information retrieval;
- query-document matching;
- cosine similarity;
- ONNX browser inference;
- Transformers.js;
- Recall@K;
- MRR@10;
- nDCG@10;
- reranking improvement;
- browser and Python latency analysis;
- Gradio;
- Vite;
- Hugging Face Static Spaces;
- Hugging Face Model Hub;
- GitHub Actions;
- automated testing;
- responsible AI;
- quality analytics positioning.

## Connection to Quality Data Science

The same architecture can support:

- finding similar GCS cases;
- retrieving related complaint descriptions;
- ranking historical root-cause investigations;
- corrective-action and CAPA search;
- supplier issue history;
- quality-document retrieval;
- future grounded quality RAG systems.

The public portfolio uses synthetic examples only. Confidential Veralto or
company data must never be uploaded to GitHub or a public Space.
