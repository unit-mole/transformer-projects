---
title: Long Document QA Longformer
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.20.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
models:
  - valhalla/longformer-base-4096-finetuned-squadv1
---

# 04 — Long-Document Question Answering with Longformer

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](#local-setup)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](#model-selection)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-orange.svg)](#gradio-application)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

A portfolio-ready Document AI application that accepts a long document, answers
a focused question with a Longformer extractive QA checkpoint, and returns:

- the predicted answer;
- an honestly labelled **model confidence proxy**;
- the supporting paragraph;
- highlighted answer evidence;
- paragraph and character-offset information;
- document length, token-window count, and inference latency.

**Live Hugging Face demo:**  
`https://huggingface.co/spaces/<YOUR_USERNAME>/long-document-qa-longformer`

**Base model:**  
`https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1`

**Free browser demo:**  
`https://huggingface.co/spaces/<YOUR_USERNAME>/long-document-qa-browser`

---

## Recommended dual-deployment architecture

Project 04 now keeps the Longformer Python implementation and adds a separate
free browser deployment baseline:

| Layer | Model and runtime | Purpose |
|---|---|---|
| GitHub project | Longformer, PyTorch, Python | Full ML engineering, evaluation, tests, notebooks, and reproducibility |
| Gradio / ZeroGPU Space | Longformer checkpoint | Primary live demonstration when compute eligibility is available |
| Static Space | DistilBERT QA ONNX through Transformers.js | Free browser demo with chunk retrieval, evidence mapping, and diagnostics |

The Static Space performs genuine Transformer inference, but it does **not**
claim to execute Longformer. It uses
`Xenova/distilbert-base-cased-distilled-squad` as a transparent browser
baseline because Longformer and BigBird are not currently listed as supported
Transformers.js architectures. See
[`PROJECT_04_DEPLOYMENT_ROADMAP.md`](PROJECT_04_DEPLOYMENT_ROADMAP.md) and
[`web/README.md`](web/README.md).

---

## Responsible-use notice

> This project is for educational and portfolio demonstration only. The model
> may return incomplete, incorrect, unsupported, or misleading answers. The
> confidence value is a model-based proxy and does not guarantee correctness.
> Highlighted evidence can be incomplete. Do not use outputs as the sole basis
> for legal, medical, financial, safety-critical, academic, official, regulated,
> or business-critical decisions. Do not upload private, confidential,
> proprietary, copyrighted, sensitive, or personally identifiable documents to
> a public demo. Human review is required.

---

## Strict project pattern

| Item | Implementation |
|---|---|
| Project number | 04 |
| Project name | `04-long-document-question-answering-longformer` |
| Application | Upload, paste, or select a long document and ask questions |
| Required outputs | Answer, confidence proxy, supporting paragraph, highlighted evidence |
| Model | Longformer QA checkpoint |
| Data | Safe synthetic reports; optional public long-document QA data |
| Evaluation | Exact Match, token F1, evidence recall, context-length analysis, latency |
| Deployment | GitHub + optional Gradio/ZeroGPU + free Static Space |

---

## Why this project matters

Long-document question answering is useful when information is buried inside
research papers, policies, manuals, reports, case histories, CAPA records,
complaint investigations, supplier reviews, or technical documents. A useful
system must do more than return a string: it should show **where the answer came
from**, communicate uncertainty, and remain stable when the document is longer
than one model input window.

This project is especially relevant to a Quality Data Scientist because the
same architecture can support safe prototypes for:

- retrieving evidence from GCS case histories;
- finding ownership and due dates in CAPA reports;
- answering questions from SOPs and technical manuals;
- locating root-cause statements in investigation records;
- searching supplier-quality reports;
- building an evidence-grounded quality knowledge base.

The public demo uses only synthetic or user-provided non-sensitive documents.

---

## What the supplied notebook originally contained

The uploaded notebook titled **LongDocQA 360**:

- generated synthetic documents;
- loaded a small SQuAD subset when available;
- repeated SQuAD contexts to imitate longer text;
- chunked by words and ranked chunks with TF-IDF;
- defaulted to a sentence-overlap heuristic;
- optionally used `distilbert-base-cased-distilled-squad`;
- produced Streamlit code and exported files.

It did **not** use Longformer by default, did not implement tokenizer-aware
long-context windows, did not map answer offsets to evidence, and did not meet
the requested Gradio/Hugging Face structure. The full audit is documented in
[`CHANGELOG_FROM_ORIGINAL.md`](CHANGELOG_FROM_ORIGINAL.md), and the original
notebook is preserved under `notebooks/archive/`.

---

## Model selection

### Selected checkpoint

```text
valhalla/longformer-base-4096-finetuned-squadv1
```

### Why it was selected

- Longformer is designed for longer sequences than standard BERT-style models.
- The checkpoint has an extractive QA head and was already fine-tuned on SQuAD v1.
- It supports approximately 4,096 tokens in one input window.
- It can be loaded directly with `AutoTokenizer` and
  `AutoModelForQuestionAnswering`.
- The base-size model is more feasible for a portfolio CPU demo than a large
  Longformer checkpoint.

### Fine-tuning honesty

This repository **does not claim that Anmol fine-tuned this model**. It uses a
published checkpoint and adds a professional inference, evidence, evaluation,
testing, and deployment layer. A future QASPER or quality-document fine-tuning
experiment can be added separately and documented with actual training records.

---

## Longformer and long-context handling

Longformer combines local sliding-window attention with selected global
attention. For question answering, this project marks question tokens for global
attention so they can interact with the full available context.

The checkpoint can process approximately 4,096 tokens, but the public CPU demo
defaults to **2,048 tokens per runtime window** to reduce memory use and latency.

For documents longer than the selected runtime window:

1. the tokenizer receives the question and complete normalized document;
2. `truncation="only_second"` preserves the question;
3. `return_overflowing_tokens=True` creates overlapping document windows;
4. `stride` controls repeated context between windows;
5. the QA model produces start and end logits for every window;
6. invalid, special-token, and overlong spans are removed;
7. the strongest valid span across all windows is selected;
8. tokenizer offsets map the answer back to the original normalized document.

This design avoids pretending that a very long document fits inside a single
model call.

---

## End-to-end workflow

```text
TXT / Markdown / CSV / PDF / pasted text
                │
                ▼
        Safe document loading
                │
                ▼
  Unicode and whitespace normalization
                │
                ▼
 Question + overlapping token windows
                │
                ▼
 Longformer extractive QA inference
                │
                ▼
 Best valid answer span across windows
                │
                ▼
 Character-offset evidence mapping
                │
                ▼
 Answer + confidence proxy + paragraph
 + highlighted evidence + diagnostics
```

---

## Document loading

Supported formats:

| Format | Behavior |
|---|---|
| `.txt` | UTF-8, UTF-8 with BOM, then Windows-1252 fallback |
| `.md` | Same text handling while preserving paragraph boundaries |
| `.csv` | Uses `text`, `content`, `document`, `context`, `paragraph`, or similar text columns |
| `.pdf` | Uses `pypdf` for selectable text |
| pasted text | Treated as an in-memory document |

Safety and reliability checks include:

- unsupported-extension rejection;
- maximum upload size;
- empty-file handling;
- no-readable-text handling;
- maximum document character limit;
- safe sample filename handling;
- clear message for scanned PDFs that require OCR.

OCR is intentionally excluded to keep the public demo lightweight and avoid
unreliable extraction.

---

## Answer extraction

The model returns start and end logits for each token. The extraction module:

- limits candidate positions to context tokens;
- searches top start and end positions;
- enforces `end >= start`;
- enforces a maximum answer-token length;
- rejects empty or invalid character offsets;
- creates one best candidate per window;
- selects the highest raw span score across windows.

The returned character offsets are used for evidence mapping and highlighting.

---

## Confidence proxy

The application calculates an uncalibrated confidence proxy using the geometric
mean of the selected start-token and end-token probabilities inside the valid
context tokens of the chosen window.

This value:

- is useful for relative inspection;
- is **not** a calibrated probability that the answer is correct;
- can be high for an incorrect answer;
- can be low for a correct answer;
- must be reviewed together with the supporting paragraph.

Very low values generate a visible warning.

---

## Supporting paragraph and highlighted evidence

The normalized document is split into paragraphs with character offsets. The
selected answer span is located inside the paragraph containing its start
offset. The application then:

- returns that paragraph;
- reports the paragraph index;
- highlights the exact answer characters;
- reports an honest message if the span cannot be mapped;
- never creates synthetic evidence for a missing span.

---

## Dataset

The committed dataset is deliberately small and safe:

- `quality_capa_report.txt`
- `supplier_quality_report.txt`
- `longformer_overview.md`
- `sample_questions.csv`
- `sample_qa_pairs.csv`

All sample documents are synthetic portfolio content. They do not contain
employer, customer, patient, proprietary, or personal data.

The evaluation schema includes:

```text
example_id
document_name
question
answer
reference_evidence
reference_paragraph_index
source_type
```

For a larger study, use a documented public dataset such as a QASPER subset,
subject to its license and practical compute requirements. Do not commit a large
or restricted full dataset.

---

## Evaluation

The code implements the required metrics.

### Exact Match

Checks whether the normalized prediction exactly equals the normalized reference
answer.

### Token-level F1

Provides partial credit for overlapping normalized answer tokens.

### Evidence recall

Checks whether the predicted supporting paragraph contains or sufficiently
covers the reference evidence.

### Context-length analysis

Groups examples into:

```text
0–512
513–1024
1025–2048
2049–4096
4097+
```

For each available bucket, the script summarizes:

- example count;
- Exact Match;
- token F1;
- evidence recall;
- average latency.

### Manual error analysis

The evaluation script creates actual examples for review, including weak answer
overlap, missed evidence, low confidence, windowing notes, and errors.

### Results policy

No metric is invented. The committed `outputs/model_metrics.json` is marked
`not_run`. Generate real results with:

```bash
python scripts/preprocess_documents.py
python scripts/evaluate_model.py
python scripts/run_context_analysis.py
```

---

## Gradio application

The Gradio interface includes:

- document upload;
- sample document selector;
- pasted text input;
- question input;
- runtime context-window slider;
- overlap/stride slider;
- answer output;
- model confidence proxy;
- supporting paragraph;
- highlighted evidence;
- document and model diagnostics;
- responsible-use disclaimer;
- model, evaluation, and limitation tabs;
- GitHub, Space, and model placeholders.

The model is lazy-loaded on the first inference request. The app does not train
a model during startup.

---

## Folder structure

```text
04-long-document-question-answering-longformer/
│
├── app.py
├── gradio_app.py
├── README.md
├── README_HUGGINGFACE.md
├── MODEL_CARD.md
├── DEPLOYMENT_HUGGINGFACE.md
├── PORTFOLIO_POSITIONING.md
├── CHANGELOG_FROM_ORIGINAL.md
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .env.example
├── pyproject.toml
├── pytest.ini
│
├── configs/
│   └── config.yaml
├── data/
│   ├── sample_documents/
│   ├── sample_questions.csv
│   ├── sample_qa_pairs.csv
│   └── README_data.md
├── notebooks/
│   ├── long_document_question_answering_longformer.ipynb
│   ├── evidence_recall_context_length_analysis.ipynb
│   └── archive/
├── src/
│   ├── config.py
│   ├── schemas.py
│   ├── data_preprocessing.py
│   ├── document_loader.py
│   ├── text_preprocessing.py
│   ├── document_chunking.py
│   ├── qa_model.py
│   ├── answer_extraction.py
│   ├── confidence_scoring.py
│   ├── evidence_highlighting.py
│   ├── inference_pipeline.py
│   ├── model_evaluation.py
│   ├── context_length_analysis.py
│   └── visualization.py
├── scripts/
│   ├── preprocess_documents.py
│   ├── evaluate_model.py
│   ├── run_context_analysis.py
│   └── run_gradio.py
├── models/
│   ├── model_metadata.json
│   ├── README.md
│   ├── long_document_qa_model/
│   └── tokenizer/
├── outputs/
├── images/
└── tests/
```

The root repository workflow is stored at:

```text
.github/workflows/04-long-document-question-answering-longformer.yml
```

---

## Local setup

### 1. Open the repository

```bash
cd transformer-models-projects
cd 04-long-document-question-answering-longformer
```

Your local parent folder may be named `transformer-projects`; Git functionality
does not depend on the Windows folder name.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Open the local Gradio URL shown in the terminal.

### 5. Run evaluation

```bash
python scripts/preprocess_documents.py
python scripts/evaluate_model.py
python scripts/run_context_analysis.py
```

### 6. Run tests

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Hugging Face Spaces deployment

1. Create a new Hugging Face Space.
2. Select **Gradio** as the SDK.
3. Use Python 3.11.
4. Place this project folder's contents at the Space repository root.
5. Confirm that `app.py`, `requirements.txt`, and `README.md` are at the root.
6. Replace `<YOUR_USERNAME>` placeholders.
7. Set optional Space variables from `.env.example`.
8. Commit and wait for the dependency build and checkpoint download.
9. Test all sample documents.
10. Add the live Space URL to this README and the root portfolio README.

See [`DEPLOYMENT_HUGGINGFACE.md`](DEPLOYMENT_HUGGINGFACE.md) for complete steps.

### Current hosting eligibility

As of July 2026, Hugging Face documentation states that Static Spaces are free
for everyone, while creating compute-backed Gradio or Docker Spaces generally
requires a paid plan. Free personal accounts in good standing may host up to two
Gradio Spaces using ZeroGPU. Because this is a Python/PyTorch Longformer app, it
requires compute and is not a pure Static Space. Check current eligibility
before deployment.

---

## Docker

```bash
docker build -t long-document-qa-longformer .
docker run --rm -p 7860:7860 long-document-qa-longformer
```

The first model download occurs when the first question is submitted.

---

## GitHub Actions

The workflow:

- runs on project pushes and pull requests;
- installs dependencies;
- compiles Python files;
- runs unit tests;
- imports the Gradio app;
- imports the inference pipeline;
- imports document loading and evidence highlighting;
- disables model loading in CI;
- avoids training and dataset downloads.

---

## Baseline comparison plan

A future experiment can compare:

| Approach | Expected role |
|---|---|
| Keyword paragraph retrieval + short QA | Simple lexical baseline |
| BM25 retrieval + short QA | Stronger retrieve-then-read baseline |
| Truncated BERT QA | Demonstrates short-context failure |
| Longformer sliding-window QA | Current long-context approach |

No baseline metric is shown until the implementations are executed on the same
evaluation examples and hardware.

---

## Limitations

- The model is extractive and cannot safely generate absent answers.
- The checkpoint is SQuAD-oriented rather than quality-domain fine-tuned.
- It lacks a reliable trained no-answer head.
- Multiple similar spans can confuse evidence selection.
- Window boundaries can reduce performance.
- 4,096-token CPU inference may be slow.
- Scanned PDFs are unsupported without OCR.
- English is the primary model language.
- Confidence is uncalibrated.
- Normalized text, rather than original binary-page coordinates, is highlighted.

---

## Future improvements

- Fine-tune Longformer or BigBird on QASPER or a safe quality-document dataset
- Add a learned no-answer threshold using SQuAD 2.0-style supervision
- Compare Longformer with BigBird and retrieve-then-read baselines
- Add BM25 or dense paragraph retrieval for very large document collections
- Add page-level PDF evidence references
- Add calibrated confidence using validation data
- Quantize or optimize the checkpoint for CPU latency
- Add multilingual long-document QA
- Add batch evaluation and experiment tracking
- Export to ONNX if model support and performance are validated

---

## Skills demonstrated

Transformer architecture, Longformer, global/local attention, extractive
question answering, long-context handling, tokenizer overflow, span extraction,
confidence communication, evidence localization, document parsing, evaluation,
error analysis, Gradio, Hugging Face Spaces, modular Python, testing, CI,
Docker, data safety, and recruiter-friendly technical documentation.

---

## Portfolio description

> Built a deployment-ready Longformer document QA system that processes
> uploaded long documents and returns grounded answer spans, a confidence proxy,
> supporting paragraphs, highlighted evidence, and context-length evaluation.

