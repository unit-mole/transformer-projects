# 06 — Multimodal Visual Question Answering Transformer

[![CI](https://github.com/unit-mole/transformer-projects/actions/workflows/06-multimodal-visual-question-answering-transformer.yml/badge.svg)](https://github.com/unit-mole/transformer-projects/actions/workflows/06-multimodal-visual-question-answering-transformer.yml)
[![Static Space](https://img.shields.io/badge/Hugging%20Face-Static%20Space-FFD21E?logo=huggingface)](https://huggingface.co/spaces/anmol-unitmole/06-multimodal-visual-question-answering-transformer)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](../LICENSE)

A portfolio-ready vision-language project that receives an image and a
natural-language question, then predicts an answer using Transformer-based
multimodal models.

## Project pattern

| Category | Implementation |
|---|---|
| Project number | 06 |
| Application | Visual Question Answering |
| User workflow | Upload image → ask question → receive answer, question type, answer type, and latency |
| Local / evaluation model | `dandelin/vilt-b32-finetuned-vqa` |
| Static browser demo model | `Xenova/moondream2` through Transformers.js |
| Dataset support | VQA v2-style subset plus safe synthetic samples |
| Evaluation | VQA consensus accuracy, exact match, category analysis, failure analysis, latency |
| Deployment | **Hugging Face Static Space** |

## Why the architecture uses two model paths

The local Python pipeline uses ViLT because it provides a compact,
classification-style VQA head and exposes answer logits for an honest
confidence proxy. The deployed Space is fully static, so it cannot run the
Python ViLT server. Its browser demo therefore uses the ONNX-compatible
Moondream2 model with Transformers.js and WebGPU.

The browser app does **not** fabricate a confidence value. It displays
`N/A` because the selected generative browser path does not expose a reliable,
calibrated confidence probability in this implementation.

## What changed from the provided notebook

The supplied notebook was a useful multimodal image–text retrieval prototype:
it generated synthetic shapes, loaded CIFAR-10 or digits, built TF-IDF and
handcrafted image features, performed similarity search, and exported a
Streamlit app. It was not a visual question-answering Transformer pipeline.

This project preserves that notebook as a legacy reference while adding:

- genuine image + question model inference;
- ViLT preprocessing and answer-logit handling;
- VQA-style consensus evaluation;
- question and answer categorization;
- latency and failure-analysis utilities;
- a browser-only vision-language demo;
- Static Space metadata and automated GitHub-to-Hugging-Face synchronization;
- tests, model card, dataset card, and privacy documentation.

## Live demo

**Planned Space:**  
`https://huggingface.co/spaces/anmol-unitmole/06-multimodal-visual-question-answering-transformer`

The first browser model download is large—approximately 1+ GB for the selected
quantized components—and requires a modern desktop browser with WebGPU. Model
files are downloaded from the Hugging Face Hub and cached by the browser.

## Responsible use and image privacy

This project is for education and portfolio demonstration only. The model can
produce incomplete, incorrect, biased, or misleading answers. Do not use it for
medical, legal, financial, security, surveillance, identity verification,
employment, insurance, or other high-stakes decisions.

Do not upload private photographs, IDs, medical images, confidential workplace
images, proprietary documents, or copyrighted images without permission.
Public sample images in this repository are synthetic and contain no people.

## Repository structure

```text
06-multimodal-visual-question-answering-transformer/
├── data/
│   ├── sample_images/
│   ├── README_data.md
│   ├── sample_questions.csv
│   └── sample_vqa_pairs.csv
├── models/
│   ├── model_metadata.json
│   └── vqa_model_reference.txt
├── notebooks/
│   ├── legacy_multimodal_image_text_understanding.ipynb
│   └── multimodal_visual_question_answering_transformer.ipynb
├── outputs/
├── scripts/
├── space/
│   ├── README.md
│   ├── index.html
│   ├── samples/
│   └── src/
├── src/vqa/
├── tests/
├── DATASET_CARD.md
├── MODEL_CARD.md
├── README_HUGGINGFACE.md
├── pyproject.toml
├── requirements-ci.txt
└── requirements.txt
```

## Local Python setup

```bash
cd 06-multimodal-visual-question-answering-transformer
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/prepare_sample_data.py
python scripts/run_local_vilt.py data/sample_images/shapes_scene.png "What color is the circle?"
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_sample_data.py
python scripts/run_local_vilt.py data/sample_images/shapes_scene.png "What color is the circle?"
```

## Evaluation

```bash
python scripts/evaluate_vqa.py --limit 3
python scripts/benchmark_latency.py --repeats 5
pytest
```

The committed output JSON files intentionally say `not_evaluated`. Replace them
only after executing the scripts on a documented dataset and hardware
configuration. Never invent portfolio metrics.

## Run the static site locally

Because the app uses JavaScript modules and browser security rules, serve it
through a local HTTP server:

```bash
cd space
python -m http.server 8000
```

Open `http://localhost:8000` in a current desktop version of Chrome or Edge.

## Deploy to a Hugging Face Static Space

1. Create a Space named `06-multimodal-visual-question-answering-transformer`.
2. Choose **Static** as the SDK and **Blank** as the template.
3. Set the Space to Public.
4. Copy the contents of the project `space/` folder into the root of the Space.
5. Confirm that the Space README contains `sdk: static` and `app_file: index.html`.
6. Wait for the Space to rebuild.
7. Open the app in Chrome or Edge and test a safe sample image.
8. Add the live URL to this README and the root repository README.

For GitHub Actions deployment, create:

- repository secret `HF_TOKEN` with Hugging Face write permission;
- repository variable `HF_SPACE_REPO` with value  
  `anmol-unitmole/06-multimodal-visual-question-answering-transformer`.

## Portfolio description

**One line:** Browser-deployed multimodal VQA system that answers
natural-language questions about images using ViLT, Moondream2,
Transformers.js, WebGPU, VQA-style evaluation, and failure analysis.

**Skills demonstrated:** multimodal AI, vision-language Transformers, VQA,
image preprocessing, question preprocessing, ViLT, browser ONNX inference,
Transformers.js, WebGPU, confidence interpretation, VQA consensus scoring,
category analysis, latency benchmarking, responsible AI, testing, CI, and
Hugging Face Static Spaces.

## Quality analytics connection

The same architecture can support future quality workflows such as asking
questions about inspection images, explaining visible defects, helping
operators review product images, and combining visual evidence with
text-based reasoning. It must still be validated carefully before any
production or safety-critical use.
