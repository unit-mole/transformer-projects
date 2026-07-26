# Project 04 Deployment Roadmap

## Static deployment does not reduce Project 04's value

Static deployment changes where inference runs; it does not turn the project
into a mock interface.

- **Python / Gradio path:** Longformer inference runs through PyTorch on a
  compute-backed environment.
- **Static browser path:** a browser-compatible extractive QA Transformer runs
  inside the visitor's browser through Transformers.js and ONNX Runtime.

The architectures must be described honestly. The browser fallback uses a
DistilBERT QA model over retrieved chunks; it does not claim to execute
Longformer.

## Best portfolio approach for Project 04

Use a four-part structure:

| Portfolio component | Purpose |
|---|---|
| GitHub repository | Complete Longformer Python project, evaluation, tests, notebooks, and engineering structure |
| Hugging Face model reference or repository | Model card, base-model attribution, dataset, metrics, limitations, and optional fine-tuned artifacts |
| Hugging Face Gradio / ZeroGPU Space | Primary live Longformer demo when eligible compute is available |
| Hugging Face Static Space | Permanently free browser QA baseline with chunking, retrieval, evidence, and diagnostics |

This structure is stronger than silently replacing Longformer with a short-text
model. It demonstrates model engineering, deployment trade-offs, and responsible
technical communication.

## 1. Keep the full Python Longformer project

Do not remove:

```text
app.py
gradio_app.py
configs/
src/
scripts/
tests/
notebooks/
outputs/
requirements.txt
MODEL_CARD.md
```

These files demonstrate:

- Longformer sparse-attention architecture;
- question-token global attention;
- overlapping tokenizer windows;
- answer-span aggregation;
- document parsing and paragraph mapping;
- confidence-proxy calculation;
- Exact Match, token F1, evidence recall, and context-length analysis;
- Gradio application engineering;
- automated tests and GitHub Actions.

## 2. Add a separate browser deployment layer

The project now includes:

```text
04-long-document-question-answering-longformer/
└── web/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── public/
    │   ├── samples/
    │   └── evaluation/
    ├── src/
    │   ├── main.js
    │   ├── config.js
    │   ├── document-parser.js
    │   ├── chunking.js
    │   ├── qa-engine.js
    │   ├── evidence.js
    │   └── styles.css
    └── tests/
```

The Static Space performs:

1. TXT, Markdown, CSV, or selectable-text PDF extraction;
2. safe text normalization;
3. overlapping long-document chunking;
4. lexical retrieval of the most relevant chunks;
5. real ONNX extractive-QA inference in the browser;
6. best-span aggregation;
7. supporting-paragraph mapping;
8. highlighted evidence and diagnostics.

## 3. Browser model disclosure

Static model:

```text
Xenova/distilbert-base-cased-distilled-squad
```

Core Python model:

```text
valhalla/longformer-base-4096-finetuned-squadv1
```

The README and interface state this difference prominently. The browser model
is a deployment baseline, not a substitute training claim.

## 4. What the Static demo displays

- full Python Longformer model name;
- browser ONNX QA model name;
- architecture comparison table;
- document upload and preloaded samples;
- extracted document preview;
- word and character counts;
- chunk size and overlap controls;
- candidate-retrieval control;
- WASM and WebGPU runtime selection;
- model-loading progress;
- answer and confidence proxy;
- supporting paragraph and highlighted evidence;
- total and evaluated chunk counts;
- browser inference latency;
- candidate-answer diagnostics;
- downloadable result JSON;
- limitations and responsible-use disclosure;
- GitHub, Gradio, Static Space, and model-card links.

## 5. Hugging Face model repository strategy

Create a personal model repository only after you have genuinely fine-tuned or
converted an artifact. Until then:

- reference the original Longformer checkpoint;
- retain the repository's `MODEL_CARD.md` for project-level documentation;
- never describe the checkpoint as trained by you;
- publish actual metrics only after running the evaluation scripts.

A future genuine repository could be named:

```text
anmol-unitmole/longformer-quality-document-qa
```

Use that name only after producing and uploading your own fine-tuned checkpoint.

## 6. Final deployment recommendation

```text
GitHub
└── Complete Longformer Python ML project

Hugging Face model page
└── Base-model attribution or genuine fine-tuned artifacts

Hugging Face Gradio / ZeroGPU Space
└── Primary Longformer demo when compute eligibility is available

Hugging Face Static Space
└── Free Transformers.js QA baseline with long-document chunk retrieval
```

## Recruiter-facing statement

> Built an evidence-grounded long-document QA system with a Longformer Python
> pipeline and a free browser deployment baseline using Transformers.js, ONNX,
> chunk retrieval, answer-span aggregation, confidence diagnostics, and
> highlighted supporting evidence.

## Official references

- Hugging Face Static Spaces: https://huggingface.co/docs/hub/spaces-sdks-static
- Hugging Face Spaces overview: https://huggingface.co/docs/hub/spaces-overview
- Transformers.js supported tasks and architectures: https://huggingface.co/docs/transformers.js/index
- Transformers.js question-answering pipelines: https://huggingface.co/docs/transformers.js/pipelines
- Browser-compatible QA model: https://huggingface.co/Xenova/distilbert-base-cased-distilled-squad
