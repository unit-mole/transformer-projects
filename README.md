# Transformer Projects

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-ffd21e.svg)](https://huggingface.co/docs/transformers/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-00a67d.svg)](https://www.sbert.net/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Browser%20Inference-005ced.svg)](https://onnxruntime.ai/)
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-6%20Live%20Spaces-2ea44f.svg)](https://huggingface.co/anmol-unitmole)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-3%20Static%20Apps-222222.svg)](https://unit-mole.github.io/transformer-projects/)
[![Vercel](https://img.shields.io/badge/Vercel-1%20RAG%20Application-black.svg)](https://10-ai-portfolio-rag-assistant-git-main-antripat.vercel.app/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Project--Specific%20CI-2088ff.svg)](https://github.com/unit-mole/transformer-projects/actions)

A structured portfolio of **ten completed Transformer projects** covering abstractive summarization, neural machine translation, retrieval and reranking, long-document question answering, instruction tuning, multimodal visual question answering, semantic search, image classification, image-text retrieval, and retrieval-augmented generation.

Each project is developed as an end-to-end case study with reproducible source code, task-appropriate evaluation, automated validation, responsible-use documentation, and a publicly accessible application.

**Portfolio status:** 10 completed and deployed projects  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)  
**Deployment portfolio:** 6 Hugging Face Spaces · 3 GitHub Pages applications · 1 Vercel application

---

## Portfolio Objective

This repository demonstrates how Transformer architectures can be applied across Natural Language Processing, Information Retrieval, Computer Vision, multimodal learning, and generative AI.

Each project is designed to move beyond notebook-only experimentation and generally contains:

- a clearly defined analytical or business problem;
- reproducible data preparation and validation;
- a Transformer architecture selected for the task;
- baseline or cross-model comparison where meaningful;
- training, validation, and test separation;
- task-appropriate evaluation metrics;
- modular source code and reusable utilities;
- saved metadata, reports, charts, and deployment artifacts;
- automated testing or GitHub Actions validation;
- an interactive public demonstration;
- responsible-use guidance;
- transparent limitations and future improvements.

The portfolio is intended to demonstrate skills relevant to:

- Data Science;
- Machine Learning;
- Applied Artificial Intelligence;
- Natural Language Processing;
- Computer Vision;
- Information Retrieval;
- Generative AI;
- Quality Analytics;
- Analytics Engineering;
- AI application development.

---

## Completed Projects

| No. | Project | Transformer Problem | Primary Deployment | Status |
|---:|---|---|---|---|
| 1 | [Abstractive Text Summarization](01-abstractive-text-summarization-transformer/) | Sequence-to-sequence abstractive summarization | Hugging Face | [Live Demo](https://huggingface.co/spaces/anmol-unitmole/01-abstractive-text-summarization-transformer) |
| 2 | [English-Hindi Neural Machine Translation](02-neural-machine-translation-transformer/) | Sequence-to-sequence translation | Hugging Face | [Live Demo](https://huggingface.co/spaces/anmol-unitmole/english-hindi-neural-machine-translation) |
| 3 | [Cross-Encoder and Bi-Encoder Ranking System](03-cross-encoder-bi-encoder-ranking-system/) | Two-stage retrieval and reranking | Hugging Face | [Live Demo](https://huggingface.co/spaces/anmol-unitmole/03-cross-encoder-bi-encoder-ranking-system) |
| 4 | [Long-Document Question Answering with Longformer](04-long-document-question-answering-longformer/) | Long-context extractive question answering | Hugging Face | [Live Demo](https://huggingface.co/spaces/anmol-unitmole/long-document-question-answering-longformer) |
| 5 | [Instruction-Tuned Domain LLM](05-instruction-tuned-domain-llm/) | FLAN-T5 instruction tuning with LoRA and PEFT | Hugging Face | [Live Demo](https://huggingface.co/spaces/anmol-unitmole/instruction-tuned-domain-llm) |
| 6 | [Multimodal Visual Question Answering](06-multimodal-visual-question-answering-transformer/) | Image-and-text reasoning | Hugging Face | [Live Demo](https://huggingface.co/spaces/anmol-unitmole/06-multimodal-visual-question-answering-transformer) |
| 7 | [Document Semantic Search with Sentence-BERT](07-document-semantic-search-sentence-bert/) | Dense document retrieval and semantic similarity | GitHub Pages | [Live Demo](https://unit-mole.github.io/transformer-projects/07-document-semantic-search-sentence-bert/) |
| 8 | [Image Classification with Vision Transformer](08-image-classification-vision-transformer/) | Vision Transformer classification and CNN comparison | GitHub Pages | [Live Demo](https://unit-mole.github.io/transformer-projects/08-image-classification-vision-transformer/) |
| 9 | [Vision-Language Image-Text Retrieval with CLIP](09-vision-language-image-text-retrieval-clip/) | Cross-modal retrieval and similarity | GitHub Pages | [Live Demo](https://unit-mole.github.io/transformer-projects/09-vision-language-image-text-retrieval-clip/) |
| 10 | [AI Portfolio RAG Assistant](10-ai-portfolio-rag-assistant/) | Retrieval-augmented portfolio assistant | Vercel | [Live Demo](https://10-ai-portfolio-rag-assistant-git-main-antripat.vercel.app/) |

---

## Portfolio at a Glance

| Coverage Area | Projects |
|---|---|
| Encoder-decoder Transformers | Summarization, translation, instruction tuning |
| Encoder-only Transformers | Ranking, semantic search, long-document QA |
| Parameter-efficient fine-tuning | FLAN-T5 LoRA and PEFT |
| Long-context processing | Longformer question answering |
| Retrieval and reranking | Bi-encoder retrieval, cross-encoder reranking, Sentence-BERT search |
| Vision Transformers | DeiT image classification |
| Multimodal Transformers | Visual question answering and CLIP retrieval |
| Retrieval-augmented generation | AI portfolio RAG assistant |
| Model conversion and browser inference | ONNX and ONNX Runtime Web |
| Human model evaluation | Instruction-tuned domain LLM |
| Static ML deployment | Hugging Face Static Spaces and GitHub Pages |
| Full-stack AI deployment | Vercel RAG application |
| Automated validation | Project-specific GitHub Actions workflows |

---

## What the Portfolio Covers

The projects are intentionally varied so that the repository demonstrates multiple Transformer families, application domains, evaluation strategies, and deployment patterns.

### Text Generation and Sequence-to-Sequence Learning

- **Abstractive Text Summarization** condenses longer documents into shorter generated summaries.
- **Neural Machine Translation** translates English input into Hindi using a text-to-text Transformer workflow.
- **Instruction-Tuned Domain LLM** adapts FLAN-T5-base to Machine Learning, Data Science, evaluation-metric, and non-confidential quality-analytics instructions.

These projects demonstrate:

- tokenization and sequence preparation;
- encoder-decoder attention;
- conditional text generation;
- decoding strategies;
- output-length control;
- sequence-to-sequence evaluation;
- generated-text quality analysis;
- responsible communication of generative limitations.

### Retrieval, Ranking, and Semantic Search

- **Cross-Encoder and Bi-Encoder Ranking System** demonstrates efficient first-stage retrieval followed by accurate reranking.
- **Document Semantic Search** converts queries and documents into dense vector representations for meaning-based retrieval.
- **AI Portfolio RAG Assistant** retrieves relevant portfolio evidence before generating an answer.

These projects demonstrate:

- sentence and document embeddings;
- cosine similarity;
- vector ranking;
- top-k retrieval;
- bi-encoder efficiency;
- cross-encoder precision;
- retrieval-quality evaluation;
- grounding and source-aware response generation.

### Long-Context Natural Language Processing

- **Long-Document Question Answering with Longformer** processes documents beyond the practical context length of standard full-attention encoders.

This project demonstrates:

- sparse local attention;
- task-specific global attention;
- long-document tokenization;
- answer-span prediction;
- context-window management;
- extractive answer evaluation;
- confidence and no-answer handling.

### Computer Vision and Multimodal Learning

- **Multimodal Visual Question Answering** combines an image and a natural-language question.
- **Image Classification with Vision Transformer** fine-tunes a DeiT-tiny Transformer and compares it with ResNet-18.
- **CLIP Image-Text Retrieval** maps images and text into a shared embedding space.

These projects demonstrate:

- image preprocessing;
- visual tokenization;
- patch embeddings;
- multimodal representations;
- image-question understanding;
- cross-modal similarity;
- zero-shot or retrieval-oriented inference;
- attention-based interpretability;
- browser-side model execution.

### Generative AI and Retrieval-Augmented Applications

- **Instruction-Tuned Domain LLM** demonstrates controlled adapter training, automated evaluation, human review, and release gating.
- **AI Portfolio RAG Assistant** demonstrates retrieval-augmented answers grounded in the portfolio's project content.

These projects demonstrate:

- instruction tuning;
- parameter-efficient fine-tuning;
- retrieval-augmented generation;
- evidence selection;
- prompt construction;
- response generation;
- limitations of automated generative metrics;
- responsible release decisions.

---

## Project Summaries

### 01 — Abstractive Text Summarization Transformer

[![Open Project 01](https://img.shields.io/badge/Open-Project%2001-2ea44f.svg)](01-abstractive-text-summarization-transformer/)
[![Live Demo](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-ffd21e.svg)](https://huggingface.co/spaces/anmol-unitmole/01-abstractive-text-summarization-transformer)

This project builds an abstractive summarization workflow that generates concise summaries rather than extracting unchanged sentences. It covers text cleaning, tokenization, sequence-length controls, generation settings, reference-based evaluation, and browser-oriented deployment.

**Key capabilities:**

- Transformer-based abstractive summarization;
- configurable input and summary lengths;
- generated-summary evaluation;
- example-driven application interface;
- reproducible preprocessing;
- deployment and CI validation.

---

### 02 — English-Hindi Neural Machine Translation

[![Open Project 02](https://img.shields.io/badge/Open-Project%2002-2ea44f.svg)](02-neural-machine-translation-transformer/)
[![Live Demo](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-ffd21e.svg)](https://huggingface.co/spaces/anmol-unitmole/english-hindi-neural-machine-translation)

This project demonstrates sequence-to-sequence translation from English to Hindi. It presents the complete translation pipeline from input normalization and tokenization through Transformer generation, evaluation, and deployment.

**Key capabilities:**

- English-to-Hindi translation;
- multilingual tokenization;
- encoder-decoder generation;
- translation-quality evaluation;
- handling of unsupported and long inputs;
- static Hugging Face deployment.

---

### 03 — Cross-Encoder and Bi-Encoder Ranking System

[![Open Project 03](https://img.shields.io/badge/Open-Project%2003-2ea44f.svg)](03-cross-encoder-bi-encoder-ranking-system/)
[![Live Demo](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-ffd21e.svg)](https://huggingface.co/spaces/anmol-unitmole/03-cross-encoder-bi-encoder-ranking-system)

This project implements a two-stage ranking system. A bi-encoder efficiently creates query and document embeddings for retrieval, while a cross-encoder jointly evaluates the query-document pairs to improve final ranking quality.

**Key capabilities:**

- dense retrieval;
- bi-encoder embeddings;
- cross-encoder reranking;
- top-k candidate selection;
- ranking-metric evaluation;
- latency and quality comparison;
- interactive ranking analysis.

---

### 04 — Long-Document Question Answering with Longformer

[![Open Project 04](https://img.shields.io/badge/Open-Project%2004-2ea44f.svg)](04-long-document-question-answering-longformer/)
[![Live Demo](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-ffd21e.svg)](https://huggingface.co/spaces/anmol-unitmole/long-document-question-answering-longformer)

This project demonstrates extractive question answering over documents that exceed the practical context length of standard Transformer encoders. Longformer's sparse attention pattern supports long inputs while preserving task-focused global attention.

**Key capabilities:**

- long-context document processing;
- sparse local attention;
- global attention for question tokens;
- extractive answer spans;
- answer confidence;
- long-input validation;
- question-answering evaluation.

---

### 05 — Instruction-Tuned Domain LLM

[![Open Project 05](https://img.shields.io/badge/Open-Project%2005-2ea44f.svg)](05-instruction-tuned-domain-llm/)
[![Live Demo](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-ffd21e.svg)](https://huggingface.co/spaces/anmol-unitmole/instruction-tuned-domain-llm)

This project instruction-tunes FLAN-T5-base using LoRA and PEFT. Two controlled experiments are trained, compared with the untouched base model, evaluated on the same 80-prompt benchmark, reviewed manually, and passed through an explicit release-quality gate.

**Key capabilities:**

- versioned instruction datasets;
- FLAN-T5 fine-tuning;
- LoRA and PEFT;
- CUDA and BF16 training;
- baseline and experiment comparison;
- ROUGE-L, BERTScore, semantic similarity, and rubric evaluation;
- bootstrap confidence intervals;
- human factuality and pairwise preference review;
- checkpoint-integrity debugging;
- responsible model-selection logic;
- interactive evaluation showcase.

**Final project finding:** Experiment 1 remained the strongest tested candidate, while Experiment 2 was preserved but not promoted after automated and human comparison.

---

### 06 — Multimodal Visual Question Answering Transformer

[![Open Project 06](https://img.shields.io/badge/Open-Project%2006-2ea44f.svg)](06-multimodal-visual-question-answering-transformer/)
[![Live Demo](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-ffd21e.svg)](https://huggingface.co/spaces/anmol-unitmole/06-multimodal-visual-question-answering-transformer)

This project combines visual and textual inputs to answer questions about images. It demonstrates multimodal preprocessing, image-question pairing, answer generation or selection, evaluation, and responsible confidence communication.

**Key capabilities:**

- image upload and validation;
- natural-language question input;
- multimodal Transformer processing;
- answer and confidence presentation;
- browser-friendly examples;
- evaluation evidence;
- limitations for ambiguous or out-of-distribution images.

---

### 07 — Document Semantic Search with Sentence-BERT

[![Open Project 07](https://img.shields.io/badge/Open-Project%2007-2ea44f.svg)](07-document-semantic-search-sentence-bert/)
[![Live Demo](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-222222.svg)](https://unit-mole.github.io/transformer-projects/07-document-semantic-search-sentence-bert/)

This project uses Sentence-BERT embeddings to retrieve documents by meaning rather than exact keyword overlap. Query and document vectors are compared using cosine similarity and ranked for interactive inspection.

**Key capabilities:**

- sentence and document embeddings;
- semantic similarity;
- top-k document retrieval;
- query-document score display;
- search-quality evaluation;
- browser-side static application;
- GitHub Pages deployment.

---

### 08 — Image Classification with Vision Transformer

[![Open Project 08](https://img.shields.io/badge/Open-Project%2008-2ea44f.svg)](08-image-classification-vision-transformer/)
[![Live Demo](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-222222.svg)](https://unit-mole.github.io/transformer-projects/08-image-classification-vision-transformer/)

This project fine-tunes a DeiT-tiny Vision Transformer for CIFAR-10 image classification, compares it with a ResNet-18 CNN baseline, exports the selected model to ONNX, validates PyTorch-to-ONNX parity, generates attention-rollout examples, and performs browser inference with ONNX Runtime Web.

**Key capabilities:**

- Vision Transformer transfer learning;
- ResNet-18 comparison;
- accuracy and macro-F1 evaluation;
- class reports and confusion matrices;
- ONNX conversion and parity validation;
- attention-rollout interpretability;
- WebGPU with WebAssembly fallback;
- private browser-side inference.

---

### 09 — Vision-Language Image-Text Retrieval with CLIP

[![Open Project 09](https://img.shields.io/badge/Open-Project%2009-2ea44f.svg)](09-vision-language-image-text-retrieval-clip/)
[![Live Demo](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-222222.svg)](https://unit-mole.github.io/transformer-projects/09-vision-language-image-text-retrieval-clip/)

This project uses CLIP to represent images and text in a shared embedding space. It supports image-to-text and text-to-image retrieval by ranking cross-modal cosine similarity scores.

**Key capabilities:**

- CLIP image and text encoders;
- shared cross-modal embeddings;
- cosine-similarity ranking;
- image-to-text retrieval;
- text-to-image retrieval;
- Recall@K-oriented evaluation;
- browser-based interactive exploration;
- static GitHub Pages deployment.

---

### 10 — AI Portfolio RAG Assistant

[![Open Project 10](https://img.shields.io/badge/Open-Project%2010-2ea44f.svg)](10-ai-portfolio-rag-assistant/)
[![Live Demo](https://img.shields.io/badge/Vercel-Live%20Demo-black.svg)](https://10-ai-portfolio-rag-assistant-git-main-antripat.vercel.app/)

This project brings the portfolio together through a retrieval-augmented assistant. It retrieves relevant project evidence before constructing a response, helping visitors navigate the repository, understand technical decisions, and identify demonstrated skills.

**Key capabilities:**

- portfolio-document ingestion;
- chunking and metadata;
- semantic retrieval;
- context selection;
- retrieval-augmented responses;
- source-aware project navigation;
- modern web application;
- Vercel deployment.

---

## Transformer Architecture Coverage

| Transformer Family | Demonstrated Through |
|---|---|
| Encoder-decoder Transformer | Summarization, translation, instruction tuning |
| Encoder-only Transformer | Longformer QA, Sentence-BERT search, ranking |
| Bi-encoder | First-stage retrieval and semantic search |
| Cross-encoder | Pairwise reranking |
| Sparse-attention Transformer | Longformer |
| Vision Transformer | DeiT-tiny image classification |
| Multimodal Transformer | Visual question answering |
| Contrastive vision-language model | CLIP |
| Parameter-efficient adapter | LoRA and PEFT |
| Retrieval-augmented generation | Portfolio RAG assistant |

---

## Evaluation Coverage

The projects use evaluation methods aligned with the actual task instead of relying on one universal metric.

| Task | Evaluation Methods |
|---|---|
| Abstractive summarization | ROUGE-oriented evaluation, summary length, qualitative review |
| Machine translation | Translation quality, sequence comparison, example analysis |
| Ranking and retrieval | Recall@K, MRR, NDCG, ranking quality, latency |
| Long-document QA | Exact match, token-level F1, answer span, confidence |
| Instruction tuning | Instruction adherence, rubric quality, ROUGE-L, BERTScore, semantic similarity, human review |
| Visual question answering | Answer accuracy, confidence, question-image analysis |
| Semantic search | Similarity quality, retrieval ranking, top-k inspection |
| Image classification | Accuracy, macro F1, class reports, confusion matrices, latency |
| CLIP retrieval | Recall@K, similarity ranking, cross-modal retrieval examples |
| RAG assistant | Retrieval relevance, grounding quality, source coverage, qualitative review |

### Why multiple evaluation methods matter

- A high similarity score does not guarantee a factually correct generated answer.
- Accuracy alone can hide class-level weaknesses.
- Retrieval quality must be evaluated at the ranks users actually inspect.
- Browser performance depends on runtime, hardware, and model size.
- Human review remains important for generated explanations and open-ended answers.
- Baseline comparison is necessary to show whether adaptation creates meaningful value.

---

## What the Repository Demonstrates

### End-to-End Transformer Delivery

The repository demonstrates the complete path from an idea to a public application:

- problem definition;
- data acquisition or creation;
- dataset validation;
- preprocessing;
- deterministic splitting;
- model configuration;
- training or transfer learning;
- baseline development;
- evaluation;
- error analysis;
- saved artifacts;
- reusable inference code;
- testing;
- CI validation;
- application development;
- public deployment;
- documentation;
- responsible-use communication.

### Model Selection Based on Evidence

The projects do not assume that the newest or largest model is automatically the best.

Examples include:

- cross-encoder quality compared with bi-encoder efficiency;
- Vision Transformer comparison against ResNet-18;
- PyTorch-to-ONNX parity validation;
- quantization candidates reviewed rather than forced into deployment;
- Experiment 1 and Experiment 2 LoRA adapters compared under the same benchmark;
- human review used when automated text metrics were insufficient;
- release gates used to block weak model promotion.

### Reliable and Reusable Engineering

The repository includes practices needed for dependable experimentation and inference:

- modular source files;
- reusable preprocessing;
- deterministic seeds;
- consistent feature and label mappings;
- safe handling of invalid inputs;
- metadata and configuration recording;
- training-history preservation;
- checkpoint and artifact verification;
- project-specific tests;
- project-specific GitHub Actions workflows;
- large-file and secret protection through `.gitignore`;
- deployment assets separated from local training environments.

### Deployment Diversity

The ten projects intentionally use three deployment approaches:

| Platform | Projects | Purpose |
|---|---:|---|
| Hugging Face Spaces | 6 | NLP, multimodal, model-evaluation, and static ML demonstrations |
| GitHub Pages | 3 | Browser-side semantic search, ONNX inference, and CLIP retrieval |
| Vercel | 1 | Full-stack retrieval-augmented portfolio assistant |

This demonstrates the ability to select a deployment method based on the application's runtime needs rather than using one platform for every project.

### Responsible Model Communication

Each project documents its scope and limitations.

The applications avoid presenting portfolio models as:

- authoritative expert systems;
- production-ready safety-critical tools;
- clinical or financial decision systems;
- guaranteed factual sources;
- universal solutions outside the evaluated data distribution.

Negative results, rejected optimization candidates, model limitations, and unsuccessful comparisons are retained where they add technical value.

---

## Repository Convention

The repository is organized as a monorepo. Each project generally follows this pattern:

```text
transformer-projects/
├── .github/
│   └── workflows/
│       ├── 01-abstractive-text-summarization-transformer.yml
│       ├── 02-neural-machine-translation-transformer.yml
│       ├── 03-cross-encoder-bi-encoder-ranking-system.yml
│       ├── 04-long-document-question-answering-longformer.yml
│       ├── 05-instruction-tuned-domain-llm.yml
│       ├── 06-multimodal-visual-question-answering-transformer.yml
│       ├── 07-document-semantic-search-sentence-bert.yml
│       ├── 08-image-classification-vision-transformer.yml
│       ├── 09-vision-language-image-text-retrieval-clip.yml
│       └── 10-ai-portfolio-rag-assistant.yml
│
├── 01-abstractive-text-summarization-transformer/
├── 02-neural-machine-translation-transformer/
├── 03-cross-encoder-bi-encoder-ranking-system/
├── 04-long-document-question-answering-longformer/
├── 05-instruction-tuned-domain-llm/
├── 06-multimodal-visual-question-answering-transformer/
├── 07-document-semantic-search-sentence-bert/
├── 08-image-classification-vision-transformer/
├── 09-vision-language-image-text-retrieval-clip/
├── 10-ai-portfolio-rag-assistant/
│
├── docs/
│   ├── 07-document-semantic-search-sentence-bert/
│   ├── 08-image-classification-vision-transformer/
│   └── 09-vision-language-image-text-retrieval-clip/
│
├── .gitignore
├── LICENSE
└── README.md
```

A typical individual project may contain:

```text
project-folder/
├── data/
├── docs/
├── images/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── deployment files
├── DATASET_CARD.md
├── MODEL_CARD.md
├── README.md
├── requirements.txt
└── supporting metadata and reports
```

The exact files differ by task, but the standards remain consistent:

- reproducible workflows;
- modular code;
- task-appropriate evaluation;
- public deployment;
- automated validation;
- safe repository practices;
- transparent limitations;
- portfolio-quality documentation.

---

## Continuous Integration

The repository uses project-specific GitHub Actions workflows rather than one oversized workflow for the entire monorepo.

Depending on the project, CI validates:

- required folder and file structure;
- Python source syntax;
- JavaScript syntax;
- JSON and JSONL validity;
- notebook JSON validity;
- pytest test suites;
- model configuration;
- static application assets;
- README image references;
- ONNX file presence and size;
- browser deployment paths;
- oversized files;
- accidental checkpoint or secret inclusion.

Project workflows run only when their relevant folders or workflow files change, keeping validation focused and efficient.

[![Open GitHub Actions](https://img.shields.io/badge/Open-GitHub%20Actions-2088ff?style=for-the-badge)](https://github.com/unit-mole/transformer-projects/actions)

---

## Deployment Directory

The GitHub Pages applications are published from the repository-level `docs/` directory:

```text
docs/
├── .nojekyll
├── 07-document-semantic-search-sentence-bert/
├── 08-image-classification-vision-transformer/
└── 09-vision-language-image-text-retrieval-clip/
```

The Hugging Face projects maintain Space-specific deployment assets inside their project folders or Hugging Face Space repositories.

The RAG assistant contains the configuration and web assets required for Vercel deployment.

---

## Run a Project Locally

Each project contains its own detailed setup instructions. The general Python workflow is:

### 1. Clone the repository

```bash
git clone https://github.com/unit-mole/transformer-projects.git
cd transformer-projects
```

### 2. Enter a project

```bash
cd 05-instruction-tuned-domain-llm
```

Replace the folder name with the project you want to run.

### 3. Create a virtual environment

**Windows**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 4. Install project dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Follow the project README

Some projects run through:

- Jupyter notebooks;
- Python scripts;
- local HTTP servers;
- static browser applications;
- Node.js development servers.

Always use the instructions in the selected project's `README.md`.

---

## Responsible Use

The repository is intended for education, experimentation, technical demonstration, and portfolio presentation.

General limitations include:

- pretrained models may contain biases inherited from their training data;
- generative models may produce unsupported or incorrect text;
- retrieval systems can miss relevant documents;
- similarity scores are not the same as factual correctness;
- confidence scores may not be calibrated probabilities;
- visual models may fail on out-of-distribution images;
- browser performance varies by device and runtime;
- benchmark results should not be generalized beyond the evaluated configuration;
- portfolio models are not automatically production-ready;
- no application should be used as the sole basis for safety-critical decisions.

Important outputs should be verified through trusted sources, domain expertise, and additional validation.

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| Abstractive generation | Project 01 |
| Neural machine translation | Project 02 |
| Dense retrieval | Projects 03, 07, and 10 |
| Cross-encoder reranking | Project 03 |
| Long-document processing | Project 04 |
| Extractive question answering | Project 04 |
| Instruction tuning | Project 05 |
| LoRA and PEFT | Project 05 |
| Human generative-model evaluation | Project 05 |
| Visual question answering | Project 06 |
| Sentence embeddings | Projects 03 and 07 |
| Vision Transformer classification | Project 08 |
| CNN versus Transformer comparison | Project 08 |
| ONNX conversion and browser inference | Project 08 |
| Attention-rollout interpretability | Project 08 |
| CLIP image-text retrieval | Project 09 |
| Retrieval-augmented generation | Project 10 |
| Static application development | Projects 01–09 |
| Full-stack AI web development | Project 10 |
| CI/CD | All ten projects |

---

## Core Skills Demonstrated

`Transformers` · `PyTorch` · `Hugging Face Transformers` · `Sentence Transformers` · `Natural Language Processing` · `Computer Vision` · `Multimodal Learning` · `Encoder-Decoder Models` · `Encoder Models` · `Self-Attention` · `Sparse Attention` · `Cross-Attention` · `Abstractive Summarization` · `Neural Machine Translation` · `Question Answering` · `Semantic Search` · `Dense Retrieval` · `Cross-Encoder Reranking` · `Sentence-BERT` · `Instruction Tuning` · `FLAN-T5` · `PEFT` · `LoRA` · `Human Model Evaluation` · `Vision Transformers` · `DeiT` · `CLIP` · `Visual Question Answering` · `Retrieval-Augmented Generation` · `ONNX` · `ONNX Runtime Web` · `WebGPU` · `JavaScript` · `HTML` · `CSS` · `Vercel` · `Hugging Face Spaces` · `GitHub Pages` · `Testing` · `GitHub Actions` · `CI/CD` · `Responsible AI Communication`

---

## Portfolio Positioning

**One-line description:** Ten end-to-end Transformer projects spanning NLP, retrieval, long-context question answering, instruction tuning, computer vision, multimodal learning, CLIP retrieval, browser inference, and retrieval-augmented generation.

**Pinned repository description:** Professional Transformer portfolio featuring ten deployed projects across summarization, translation, semantic retrieval, reranking, Longformer QA, FLAN-T5 LoRA instruction tuning, multimodal VQA, Vision Transformers, CLIP, and RAG—with rigorous evaluation, project-specific CI, Hugging Face Spaces, GitHub Pages, and Vercel deployment.

This portfolio connects naturally to a Quality Data Scientist background because Transformer systems can support:

- technical-document summarization;
- multilingual communication;
- knowledge retrieval;
- long-document analysis;
- analytics education;
- non-confidential quality-data workflows;
- visual inspection support;
- image and document search;
- evidence-grounded AI assistants;
- structured model evaluation and release governance.

---

## License

This repository is distributed under the [MIT License](LICENSE).

Individual models, datasets, and third-party libraries remain subject to their original licenses and usage conditions.

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Natural Language Processing | Computer Vision | Analytics Engineering | Quality Analytics
