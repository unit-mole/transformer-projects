---
title: English Hindi Neural Machine Translation
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.20.0
python_version: 3.11
app_file: app.py
pinned: false
license: mit
suggested_hardware: cpu-basic
---

# English–Hindi Neural Machine Translation with Transformers

[![Project](https://img.shields.io/badge/Project-02%20of%2010-blue)](#)
[![Task](https://img.shields.io/badge/Task-Neural%20Machine%20Translation-green)](#)
[![Models](https://img.shields.io/badge/Models-MarianMT-orange)](#)
[![Demo](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-yellow)](<YOUR_HUGGINGFACE_SPACE_URL>)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black)](#)

A production-style portfolio project for **English-to-Hindi and Hindi-to-English neural machine translation** using lightweight directional MarianMT encoder-decoder Transformers. The project includes automatic language detection, sentence translation, CSV batch translation, confidence-proxy scoring, SacreBLEU and chrF evaluation, latency tracking, manual error analysis, tests, CI, Docker, and a Gradio interface prepared for Hugging Face Spaces.

## Responsible-use notice

> This project is for educational and portfolio demonstration purposes only. Generated translations may be incomplete, inaccurate, culturally inappropriate, or may miss important context. The confidence score is a model-based proxy and does not guarantee translation correctness. Do not use generated translations as the sole basis for legal, medical, financial, immigration, safety-critical, academic, official, or business-critical decisions. Do not paste private, confidential, sensitive, copyrighted, or personally identifiable text into a public demo. Human review is required before real-world use.

## Strict project pattern

| Field | Implementation |
|---|---|
| Project number | 02 |
| Project name | `02-neural-machine-translation-transformer` |
| Application | English-to-Hindi and Hindi-to-English translation |
| Required features | Automatic language selection, sentence translation, batch translation, confidence information |
| Models | `Helsinki-NLP/opus-mt-en-hi` and `Helsinki-NLP/opus-mt-hi-en` |
| Dataset | IIT Bombay English–Hindi Parallel Corpus support plus safe sample CSV files |
| Evaluation | SacreBLEU, chrF, latency, direction-wise examples, manual error analysis |
| Deployment | Hugging Face Spaces with Gradio |

## Why this project is portfolio-worthy

The project demonstrates more than calling a translation pipeline:

- direct encoder-decoder model loading instead of a deprecated high-level pipeline;
- lazy loading of two directional models so application import never triggers model downloads;
- Unicode-safe Hindi and Devanagari preprocessing;
- script-ratio language detection with mixed/uncertain handling;
- sentence and batch inference through the same reusable pipeline;
- an honest confidence proxy derived from normalized generation scores;
- reproducible evaluation without invented metrics;
- modular code, unit tests, CI, model metadata, deployment documentation, and responsible-use controls.

## Model selection

| Direction | Model | Reason |
|---|---|---|
| English → Hindi | `Helsinki-NLP/opus-mt-en-hi` | Lightweight Marian encoder-decoder model built for the exact direction |
| Hindi → English | `Helsinki-NLP/opus-mt-hi-en` | Separate directional Marian model with direct Hindi-to-English support |

Both are loaded with `AutoTokenizer` and `AutoModelForSeq2SeqLM`. The app does **not** train at startup. Models are downloaded from the Hugging Face Hub on the first request for each direction and cached by the runtime.

### Generation configuration

- Beam search: 4 beams
- Maximum source length: 512 tokens
- Maximum new target tokens: 256
- Early stopping: enabled
- Repetition control: 3-gram blocking
- Logit renormalization: enabled

All values are stored in `models/model_metadata.json` and `configs/model_config.yaml`.

## Dataset

The project supports the public `cfilt/iitb-english-hindi` dataset, whose rows expose a `translation` object containing English (`en`) and Hindi (`hi`) text. The full corpus is intentionally not committed to GitHub. Use `src/dataset_loader.py` to load a bounded split or use your own permitted parallel CSV.

Included safe samples:

- `data/sample_translation_pairs.csv`
- `data/sample_batch_translation.csv`

See [data/README_data.md](./data/README_data.md) for licensing, storage, and column guidance.

## Architecture

```text
User text / CSV
        │
        ▼
Unicode-safe preprocessing
        │
        ▼
Script-based language detection
        │
        ├── English → EN→HI MarianMT
        ├── Hindi   → HI→EN MarianMT
        └── Mixed/uncertain → request manual direction
        │
        ▼
Beam-search generation
        │
        ├── translated text
        ├── confidence proxy
        ├── detected language
        └── latency
```

## Automatic language selection

`src/language_detection.py` counts Latin and Devanagari alphabetic characters.

- predominantly Latin text → English;
- predominantly Devanagari text → Hindi;
- meaningful presence of both scripts → mixed;
- too little alphabetic evidence → uncertain.

Mixed or uncertain inputs never crash the app. The user receives a clear message and can choose a manual direction.

## Confidence proxy

The confidence displayed by the application is **not a calibrated probability of correctness**. When beam-search sequence scores are available, the system transforms the normalized log-sequence score into a bounded proxy. A documented heuristic fallback uses output length, repetition, and unknown-token behavior.

The app always labels this value as a **confidence proxy** and includes the method used.

## Evaluation

Run:

```bash
python scripts/evaluate_model.py \
  --input data/sample_translation_pairs.csv \
  --output-dir outputs/generated
```

The script records:

- SacreBLEU;
- chrF;
- average, minimum, and maximum latency;
- latency by direction;
- example-level predictions;
- model and evaluation metadata.

The committed JSON files under `outputs/` contain `null` metric values until evaluation is actually run. This avoids presenting fabricated results.

### Why both SacreBLEU and chrF?

- **SacreBLEU** standardizes BLEU reporting and measures word/n-gram overlap.
- **chrF** measures character n-gram overlap and is useful for morphologically rich languages and spelling-level variation.
- **Latency** shows practical deployment performance.
- **Manual error analysis** catches named-entity, number, gender, tense, word-order, under-translation, and over-translation errors that overlap metrics can miss.

## Gradio demo

The interface provides:

1. sentence translation;
2. automatic or manual direction selection;
3. detected language and direction;
4. confidence proxy and latency;
5. CSV upload and text-column selection;
6. translated preview and downloadable CSV;
7. model details, evaluation guidance, risks, and limitations.

### Live links

- Hugging Face Space: `<YOUR_HUGGINGFACE_SPACE_URL>`
- Hugging Face model repository, when fine-tuned: `<YOUR_HUGGINGFACE_MODEL_URL>`
- GitHub repository: `<YOUR_GITHUB_REPOSITORY_URL>`

## Local setup

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd transformer-projects/02-neural-machine-translation-transformer

python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL printed in the terminal. The first translation in each direction downloads its model.

## Batch translation

```bash
python scripts/run_batch_translation.py \
  --input data/sample_batch_translation.csv \
  --text-column text \
  --direction auto \
  --output outputs/generated/batch_translations.csv
```

Output columns:

```text
original_text
detected_language
translation_direction
translated_text
confidence_score
confidence_method
latency_seconds
status
error
```

## Optional dataset preparation

```bash
python -c "from src.dataset_loader import load_iitb_dataframe; print(load_iitb_dataframe('validation').head())"
```

For a large experiment, always use a bounded split first and confirm the corpus terms before redistribution.

## Optional fine-tuning

The default demo uses pretrained directional MarianMT models. Fine-tuning is optional:

```bash
python scripts/train_model.py \
  --direction en_hi \
  --dataset cfilt/iitb-english-hindi \
  --train-split "train[:10000]" \
  --validation-split "validation" \
  --output-dir models/fine_tuned_en_hi
```

Fine-tuned artifacts should be pushed to a dedicated Hugging Face model repository or tracked with Git LFS, not committed as normal Git blobs.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests use mock translation engines and do not download model weights.

## Hugging Face Spaces deployment

1. Create a new Space.
2. Select **Gradio**.
3. Copy the contents of this project folder into the Space repository root.
4. Keep `app.py`, `requirements.txt`, and `README.md` at the root.
5. Commit and wait for the Space to build.
6. Test both directions and the CSV workflow.
7. Replace the live-link placeholders in this README and in the main repository README.

Detailed instructions: [DEPLOYMENT_HUGGINGFACE.md](./DEPLOYMENT_HUGGINGFACE.md).

> Current hosting note: Gradio Spaces use hosted compute. Account-plan or ZeroGPU eligibility may affect whether a new Gradio Space can be created without cost. The application itself is CPU-compatible and does not require a GPU.

## Folder structure

```text
02-neural-machine-translation-transformer/
├── app.py
├── gradio_app.py
├── configs/
├── data/
├── images/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── CHANGELOG_FROM_ORIGINAL.md
├── DEPLOYMENT_HUGGINGFACE.md
├── MODEL_CARD.md
├── README_HUGGINGFACE.md
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── .dockerignore
```

## Skills demonstrated

Transformer encoder-decoder architecture, MarianMT, multilingual NLP, English–Hindi translation, Devanagari handling, language detection, generation scoring, batch inference, SacreBLEU, chrF, latency analysis, manual error analysis, Gradio, Hugging Face Spaces, modular Python, testing, CI/CD, Docker, model cards, responsible AI, and recruiter-friendly documentation.

## Quality Data Scientist relevance

The same architecture can support multilingual customer complaints, service-case notes, quality reports, global product feedback, issue descriptions, and business communication workflows. The project connects practical quality analytics experience with deployable NLP and applied AI engineering.

## Portfolio one-liner

> Built a bidirectional English–Hindi MarianMT translation system with automatic language detection, CSV batch inference, confidence-proxy scoring, SacreBLEU/chrF evaluation, Gradio, automated tests, and Hugging Face deployment readiness.

## Limitations

- General-purpose translation can fail on technical, legal, medical, cultural, or domain-specific language.
- Named entities, numbers, gender, honorifics, idioms, and mixed-language text require careful review.
- Automatic metrics do not guarantee semantic or factual correctness.
- CPU inference can be slow during cold starts.
- The confidence proxy is not calibrated.
- Public demos must not receive sensitive text.

## Future improvements

- fine-tune on a carefully licensed, cleaned domain subset;
- add COMET or human preference evaluation;
- calibrate confidence against human judgments;
- add glossary-constrained decoding;
- quantize models for lower latency;
- add document translation with sentence alignment;
- add a browser-only fallback when practical.
