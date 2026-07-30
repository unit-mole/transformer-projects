# 10 — AI Portfolio RAG Assistant

A full-stack, Vercel-ready **Transformer Retrieval-Augmented Generation capstone** that searches Anmol Tripathi's verified public AI portfolio, generates evidence-grounded answers, and exposes source citations, retrieval scores, evaluation results, and latency.

[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](#)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue?logo=typescript)](#)
[![Transformer](https://img.shields.io/badge/Retriever-MiniLM%20%7C%20E5-orange)](#transformer-models)
[![Vercel](https://img.shields.io/badge/Deploy-Vercel-black?logo=vercel)](#vercel-deployment)
[![Evaluation](https://img.shields.io/badge/Evaluation-Retrieval%20%7C%20Grounding%20%7C%20Citations-green)](#evaluation)

> **Responsible use:** The assistant answers only from indexed public portfolio documentation. It is not an official resume, employment verification, professional reference, or source of confidential company information. Never index Hach/GCS files, internal quality data, private email, customer or supplier records, credentials, proprietary documents, or PII. Review generated answers before official use.

## Live demo

- **Vercel:** `https://YOUR-PROJECT.vercel.app`
- **Health:** `https://YOUR-PROJECT.vercel.app/api/health`
- **Evaluation:** `https://YOUR-PROJECT.vercel.app/api/evaluation`

## Project pattern

| Field | Final implementation |
|---|---|
| Project number | 10 |
| Application | RAG assistant over ANN, Simple RNN, LSTM, BiLSTM, CNN, and Transformer documentation |
| Retriever | `sentence-transformers/all-MiniLM-L6-v2`; optional E5-small comparison and MiniLM cross-encoder reranking |
| Generator | Small instruction Transformer through Hugging Face; FLAN-T5 local evaluation; grounded extractive fallback |
| Vector store | Normalized, precomputed static JSON embeddings |
| Metrics | Hit Rate@K, Precision@K, Recall@K, MRR, MAP, nDCG@K, groundedness, citation precision/completeness, refusal accuracy, latency |
| Deployment | Next.js serverless routes on Vercel |

## Why this is a genuine Transformer project

The final workflow generates document embeddings with a Sentence Transformer and embeds each user question with the same model. Semantic retrieval is therefore performed in a learned Transformer embedding space—not with the starter hash vectors. An instruction-tuned Transformer can then compose an answer from the retrieved evidence. The deterministic extractive composer remains available as a safe fallback.

The checked-in starter artifacts are intentionally marked as incomplete until the GPU notebook is run. Do not describe the starter hash mode as the final Transformer implementation.

## Architecture

```text
Public GitHub portfolio Markdown
        ↓
Safe collection + duplicate detection
        ↓
Markdown cleaning + section-aware chunking
        ↓
MiniLM / E5 document embeddings on local RTX GPU
        ↓
Normalized static JSON vector store
        ↓
Vercel Next.js API
        ↓
Question embedding + hybrid vector/lexical retrieval
        ↓
Optional cross-encoder reranking
        ↓
Grounded instruction-model answer
        ↓
[S#] citations + evidence cards + latency + warnings
```

## Transformer models

### Retriever

Primary model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It produces normalized 384-dimensional document/query vectors suitable for compact static deployment. The evaluation pipeline can also compare:

```text
intfloat/e5-small-v2
cross-encoder/ms-marco-MiniLM-L6-v2
```

### Generator

Offline evaluation supports:

```text
google/flan-t5-base
```

The deployed app supports a server-side Hugging Face instruction model configured through environment variables. The app never exposes API tokens in browser code.

### Groundedness evaluator

Claim support is measured with:

```text
cross-encoder/nli-deberta-v3-small
```

Each answer claim is compared with its cited evidence. This automated score is clearly labeled as an NLI-based evaluation and can be supplemented with human review.

## Portfolio corpus

The complete corpus should include public documents from:

- ANN / deep-learning projects
- Simple RNN projects
- LSTM projects
- Bidirectional LSTM projects
- CNN / computer-vision projects
- Transformer projects 01–10
- Model cards, dataset cards, deployment guides, and verified evaluation summaries

Each chunk stores project, repository, source file, heading, path, checksum, category, deployment, keywords, word boundaries, and source URL.

## Chunking strategy

Markdown is split by section and then into overlapping word windows. Default settings are 220 words with 50-word overlap. Headings and technical terms are preserved so that model names, datasets, metrics, and deployment details remain searchable.

## Retrieval

The Vercel runtime combines:

```text
hybrid score = semantic weight × cosine similarity
             + lexical weight × query-token coverage
```

Filters are supported for project category, deployment platform, and project ID. Weak results are rejected by the grounded fallback rather than converted into confident unsupported answers.

## Source citations

Every retrieved source receives `[S1]`, `[S2]`, and so on. Citation cards expose:

- project name and ID
- source file and section
- chunk ID
- evidence excerpt
- semantic, lexical, and combined relevance
- source path and repository URL

The instruction prompt requires a citation after each factual claim. Unsupported questions must return the documented refusal sentence.

## Evaluation

The evaluation notebook uses at least 40 curated questions, including factual, paraphrased, cross-project, deployment, metric, ambiguous, and unsupported cases.

### Retrieval metrics

| Metric | Interpretation |
|---|---|
| Hit Rate@K | Whether at least one relevant project is retrieved |
| Precision@K | Fraction of top-K retrieved projects that are relevant |
| Recall@K | Fraction of all expected relevant projects retrieved |
| MRR | Rank of the first relevant project |
| MAP@K | Precision across relevant ranks |
| nDCG@K | Ranking quality with position discounting |

This corrects the earlier starter calculation that treated any relevant hit as full recall.

### Answer metrics

- Claim-level NLI groundedness
- Citation precision
- Citation completeness
- Unsupported-claim rate
- Unsupported-question refusal accuracy
- Local embedding, retrieval, generation, total, median, P90, and P95 latency
- Deployed Vercel wall-clock latency
- Manual error analysis

No metric is presented as final until it is generated from the complete corpus and committed evaluation artifacts.

## Recommended quality gates

| Gate | Target |
|---|---:|
| Portfolio category coverage | 6/6 |
| Evaluation set | ≥ 40 questions |
| Recall@5 | ≥ 0.80 |
| nDCG@5 | ≥ 0.75 |
| Groundedness | ≥ 0.85 |
| Citation precision | ≥ 0.85 |
| Citation completeness | ≥ 0.85 |
| Refusal accuracy | ≥ 0.80 |

These are portfolio targets, not fabricated results. A failed gate is a signal to improve data, chunking, retrieval, reranking, or prompting.

## GPU notebook workflow

The main reproducible notebook is:

```text
notebooks/01-build-and-evaluate-transformer-rag.ipynb
```

It performs GPU diagnostics, corpus preparation, MiniLM/E5 embedding generation, retrieval comparisons, instruction-model evaluation, groundedness/citation scoring, latency benchmarking, chart creation, tests, and quality-gate reporting.

See [PROJECT_10_GPU_RUNBOOK.md](PROJECT_10_GPU_RUNBOOK.md) for Windows/RTX steps.

## Quick local setup

```bash
npm install
npm run validate:data
npm run dev
```

For the complete offline evaluation:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements-evaluation.txt
python scripts/collect_github_portfolio_docs.py --clean
python scripts/run_full_evaluation.py --device cuda --generator flan-t5-base --include-e5 --include-reranker
npm run validate:data
npm test
npm run typecheck
npm run build
```

Install the CUDA-enabled PyTorch build for the local machine before installing the evaluation requirements.

## Generated artifacts

```text
public/data/document_chunks.json
public/data/embeddings.json
public/data/metadata.json
public/data/evaluation_questions.json
public/data/evaluation_summary.json
outputs/retrieval_benchmark.json
outputs/answer_groundedness_results.json
outputs/citation_correctness_results.json
outputs/response_latency_results.json
outputs/deployed_latency_results.json
outputs/rag_answer_examples.csv
outputs/retrieval_method_comparison.png
outputs/rag_answer_quality.png
outputs/response_latency_distribution.png
```

Only commit measured outputs. Never manually replace pending values with estimated numbers.

## API routes

### `POST /api/chat`

Returns the grounded answer, citations, retrieved chunks, runtime modes, support warning, model/corpus metadata, and latency.

### `POST /api/retrieve`

Returns ranked source chunks without generation.

### `GET /api/health`

Checks artifact loading and reports whether real Transformer embeddings are active.

### `GET /api/evaluation`

Returns the committed evaluation summary displayed by the UI.

## Vercel deployment

Import `unit-mole/transformer-projects` into Vercel and select:

```text
Root Directory: 10-ai-portfolio-rag-assistant
Framework: Next.js
Build command: npm run build
Install command: npm install
```

Configure the same embedding model used to create `public/data/embeddings.json`:

```text
HF_API_TOKEN=...
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Optional generation:

```text
USE_HF_GENERATOR=true
HF_GENERATOR_MODEL=your-supported-instruction-model
```

See [README_VERCEL.md](README_VERCEL.md).

## Folder structure

```text
10-ai-portfolio-rag-assistant/
├── app/                         # Next.js UI and API routes
├── components/                  # Assistant, citations, metrics, evaluation UI
├── config/                      # Public repository collection config
├── data/                        # Raw safe corpus and processed artifacts
├── lib/                         # Vercel retrieval/generation implementation
├── notebooks/                   # End-to-end GPU evaluation notebook
├── outputs/                     # Measured evaluation JSON/CSV/PNG files
├── public/data/                 # Runtime vector store and summary
├── scripts/                     # Reproducible pipeline commands
├── src/                         # Python data, retrieval, generation, evaluation modules
├── tests/ and tests-ts/         # Python and Node validation
├── PROJECT_10_GPU_RUNBOOK.md
├── MODEL_CARD.md
├── DATASET_CARD.md
└── README_VERCEL.md
```

## Portfolio positioning

**One-line description:** Built and deployed an evaluated Transformer RAG assistant that semantically searches a multi-repository AI portfolio, generates grounded answers, and exposes claim-level citations, retrieval quality, and latency.

**Skills demonstrated:** Transformer embeddings, semantic search, RAG, cross-encoder reranking, instruction models, NLI evaluation, source attribution, information retrieval metrics, Python pipelines, GPU inference, Next.js, TypeScript, Vercel serverless APIs, testing, and CI.

The architecture also relates naturally to trusted enterprise search over approved quality documentation, root-cause knowledge, SOPs, technical manuals, CAPA records, and analytics documentation. This public demo must never contain internal business data.

## Current honesty label

The repository is **9/10-capable**, not automatically 9/10 merely because the code exists. It becomes a defensible 9/10 portfolio capstone after the complete public corpus is indexed, real Transformer artifacts replace the starter hash vectors, the notebook passes its quality gates, the Vercel build succeeds, and the committed README shows actual measured results.

## License

Code is MIT-licensed. Source documents, models, and datasets remain subject to their own licenses and usage terms.
