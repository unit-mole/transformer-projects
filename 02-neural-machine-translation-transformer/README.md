# Project 02 — English–Hindi Neural Machine Translation Transformer

[![Project](https://img.shields.io/badge/Project-02%20of%2010-blue)](#)
[![Task](https://img.shields.io/badge/Task-Neural%20Machine%20Translation-green)](#)
[![Models](https://img.shields.io/badge/Models-MarianMT-orange)](#)
[![Deployment](https://img.shields.io/badge/Deployment-Hugging%20Face%20Static%20Space-yellow)](#)
[![Runtime](https://img.shields.io/badge/Browser-Transformers.js-purple)](#)

A professional portfolio project for **English-to-Hindi and Hindi-to-English
neural machine translation** using MarianMT encoder-decoder Transformers. It now
combines a complete Python ML engineering implementation with a free browser
application that performs real ONNX Transformer inference on a Hugging Face
Static Space.

## Portfolio deployment strategy

| Layer | What it proves |
|---|---|
| GitHub project | Python, PyTorch, Hugging Face Transformers, preprocessing, testing, evaluation, notebooks and CI |
| Hugging Face model references | Model lineage, base models, ONNX conversions and model cards |
| Hugging Face Static Space | Live browser inference, user experience, latency, tokenization, batch translation and free deployment |

See [PORTFOLIO_DEPLOYMENT_MAP.md](./PORTFOLIO_DEPLOYMENT_MAP.md).

## Project pattern

| Field | Implementation |
|---|---|
| Project | `02-neural-machine-translation-transformer` |
| Application | English-to-Hindi and Hindi-to-English translation |
| Features | Automatic language selection, sentence translation, batch CSV translation, confidence information |
| Python models | `Helsinki-NLP/opus-mt-en-hi`, `Helsinki-NLP/opus-mt-hi-en` |
| Browser models | `Xenova/opus-mt-en-hi`, `Xenova/opus-mt-hi-en` |
| Dataset support | IIT Bombay English–Hindi corpus and safe sample CSV files |
| Evaluation | SacreBLEU, chrF, latency and manual error analysis |
| Public deployment | Free Hugging Face Static Space using Transformers.js |

## What changed for the free Static Space

The original Python/Gradio application remains in the repository as a local
engineering interface. A separate `web/` deployment layer was added because a
Static Space cannot execute Python, PyTorch or Gradio.

The browser app performs **real MarianMT inference**, not simulated output:

```text
User text
  → Unicode and script-based language detection
  → SentencePiece tokenization
  → Quantized Marian encoder-decoder ONNX model
  → Beam-search translation
  → translated text + tokens + latency + confidence proxy
```

The selected directional model is downloaded lazily and cached by the browser.
The app first requests q4 weights and falls back to q8 when needed.

## Key portfolio features

- Separate EN→HI and HI→EN encoder-decoder Transformer models
- Unicode-safe Hindi and Devanagari handling
- Automatic language detection with mixed/uncertain safeguards
- Python sentence and CSV batch inference
- Browser sentence and CSV batch inference
- Source and target token preview in the Static demo
- Beam-size and maximum-token generation controls
- Confidence-proxy scoring with transparent heuristic explanation
- SacreBLEU, chrF, latency and direction-wise evaluation
- Manual error analysis for entities, numbers, grammar and domain terms
- GitHub Actions for Python tests and Static frontend validation
- Responsible-use controls and model lineage documentation

## Repository structure

```text
02-neural-machine-translation-transformer/
├── app.py                         # Local Gradio/Python entry point
├── gradio_app.py                  # Local engineering interface
├── configs/
├── data/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/                           # Python ML pipeline
├── tests/                         # Python tests
├── web/                           # Free Hugging Face Static Space
│   ├── README.md                  # Static Space metadata
│   ├── index.html
│   ├── package.json               # Dependency-free Node validation scripts
│   ├── assets/
│   ├── data/
│   ├── src/
│   │   ├── main.js
│   │   ├── translation.worker.js
│   │   ├── language-detection.js
│   │   ├── confidence.js
│   │   ├── csv.js
│   │   └── styles.css
│   └── tests/                     # Browser utility unit tests
├── DEPLOYMENT_HUGGINGFACE_STATIC.md
├── PORTFOLIO_DEPLOYMENT_MAP.md
├── MODEL_CARD.md
├── requirements.txt
├── requirements-ci.txt
└── README.md
```

## Run the Python project locally

```bash
cd 02-neural-machine-translation-transformer
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

The Python app downloads the Helsinki-NLP models only when a direction is first
used. It does not train during startup.

## Run the Static frontend locally

A local HTTP server is required because browser workers and ES modules should
not be opened through `file://`.

```bash
cd web
python -m http.server 8000
```

Open `http://localhost:8000`.

Run dependency-free frontend tests:

```bash
cd web
npm run validate
```

## Evaluate the Python models

```bash
python scripts/evaluate_model.py   --input data/sample_translation_pairs.csv   --output-dir outputs/generated
```

The project never invents metrics. Committed placeholders remain null until a
real evaluation is executed. After evaluation, copy verified values into
`web/data/evaluation-results.json` so the Static Space displays them.

## Deploy free on Hugging Face

Follow [DEPLOYMENT_HUGGINGFACE_STATIC.md](./DEPLOYMENT_HUGGINGFACE_STATIC.md).
Upload the **contents of `web/`** to a Static Space. No paid Gradio or Docker
plan is required, and no inference API key is used.

## Responsible use

This project is for education and portfolio demonstration. Translations may be
incorrect, incomplete, culturally inappropriate or contextually misleading.
The confidence proxy is not a calibrated probability. Do not submit private,
confidential, legal, medical, immigration, safety-critical or business-critical
text. Human review is required before real-world use.

## Skills demonstrated

Transformer architecture, MarianMT, multilingual NLP, English–Hindi translation,
PyTorch, Hugging Face Transformers, Transformers.js, ONNX Runtime Web, browser
workers, Unicode handling, SacreBLEU, chrF, latency analysis, batch processing,
unit testing, CI, responsible AI and free static deployment.


## Portfolio-Grade Fine-Tuning and Evaluation

Project 02 includes a dedicated RTX GPU workflow that compares the original pretrained MarianMT models with direction-specific fine-tuned models on a deterministic held-out IIT Bombay English–Hindi test subset. The workflow reports SacreBLEU with its reproducibility signature, chrF, chrF++, TER, latency percentiles, throughput, GPU memory, preservation diagnostics, bootstrap confidence intervals, paired significance results, and human error analysis.

Run:

```cmd
jupyter lab notebooks/03_portfolio_grade_marianmt_finetuning_evaluation.ipynb
```

See [`EVALUATION_WORKFLOW.md`](EVALUATION_WORKFLOW.md) for the complete process. Result files remain truthful placeholders until the notebook is executed locally. The Static Space continues to use pretrained quantized ONNX models until the fine-tuned checkpoints are converted in a later deployment step.
