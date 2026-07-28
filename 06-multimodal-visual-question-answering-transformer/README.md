# 06 — Multimodal Visual Question Answering Transformer

[![CI](https://github.com/unit-mole/transformer-projects/actions/workflows/06-multimodal-visual-question-answering-transformer.yml/badge.svg)](https://github.com/unit-mole/transformer-projects/actions/workflows/06-multimodal-visual-question-answering-transformer.yml)
[![Static Space](https://img.shields.io/badge/Hugging%20Face-Static%20Space-FFD21E?logo=huggingface)](https://huggingface.co/spaces/anmol-unitmole/06-multimodal-visual-question-answering-transformer)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](../LICENSE)

A portfolio-ready multimodal Transformer project that accepts an image and a
natural-language question, generates an answer in the browser, reports an
honest token-likelihood diagnostic, and includes a balanced 60-pair evaluation
lab.

## Project pattern

| Category | Implementation |
|---|---|
| Project number | 06 |
| Application | Multimodal Visual Question Answering |
| User workflow | Select image → ask question → receive answer, confidence diagnostic, type, and latency |
| Static browser model | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| Local Python reference model | `dandelin/vilt-b32-finetuned-vqa` |
| Browser runtime | Transformers.js + ONNX Runtime Web + WebGPU |
| Evaluation suite | 60 synthetic image-question pairs, balanced across 6 categories |
| Deployment | Hugging Face **Static Space** |

## Live demo

**Space page:**
`https://huggingface.co/spaces/anmol-unitmole/06-multimodal-visual-question-answering-transformer`

The first browser model download can take several minutes. Later runs reuse the
browser cache. Use a current desktop version of Chrome or Edge with WebGPU.

## Browser answer-confidence design

SmolVLM is a generative vision-language model. It does not provide a calibrated
probability that an answer is factually correct.

The website therefore uses the recruiter-friendly label:

```text
Answer confidence
Not available for this generative model
```

with this explanation:

```text
Token-generation scores are not calibrated as probabilities of factual correctness.
```

When Transformers.js returns per-token generation scores, the app calculates a
**generation confidence proxy** using the geometric mean of the selected-token
probabilities. The displayed percentage is a decoding diagnostic only. It is
not an accuracy probability, must not be used for high-stakes decisions, and
must not be described as calibrated confidence.

A genuinely calibrated confidence estimate would require a held-out labeled
dataset, correctness labels, a calibration method such as temperature scaling
or isotonic regression, and calibration metrics such as expected calibration
error and Brier score.

## Built-in 60-pair browser evaluation

The Static Space includes an evaluation lab containing:

- 10 color questions;
- 10 object-identification questions;
- 10 counting questions;
- 10 yes/no questions;
- 10 action or scene questions;
- 10 spatial-relation questions.

The browser runs all 60 records through the same SmolVLM WebGPU pipeline and
calculates:

- overall accepted-answer accuracy;
- category-wise accuracy;
- answer failure rate;
- average inference latency;
- minimum and maximum inference latency;
- a downloadable JSON report;
- a manual failure-analysis preview.

This is a synthetic portfolio benchmark, not an official VQA v2 score. Do not
publish numerical claims until the complete evaluation has been run on the
intended browser and hardware.

## Architecture

The project intentionally maintains two model paths:

1. **Static browser demo:** SmolVLM-256M-Instruct runs entirely in the visitor's
   browser using Transformers.js, ONNX Runtime Web, and WebGPU.
2. **Local Python reference:** ViLT supports reproducible classification-style
   VQA inference, answer logits, and additional evaluation experiments.

No training occurs when the Static Space starts, and model weights are not
committed to GitHub.

## What changed from the supplied notebook

The supplied notebook was a multimodal image-text retrieval prototype based on
synthetic images, TF-IDF, handcrafted visual features, similarity search, and a
Streamlit export. This project preserves it as a legacy reference and adds:

- genuine image-plus-question Transformer inference;
- SmolVLM browser inference with WebGPU;
- ViLT local reference inference;
- non-empty prediction, confidence, type, latency, and failure states;
- generation-score confidence diagnostics;
- a 60-pair balanced evaluation dataset;
- browser evaluation, category analysis, latency analysis, and JSON export;
- tests, CI, model card, dataset card, privacy guidance, and Static Space deployment.

## Repository structure

```text
06-multimodal-visual-question-answering-transformer/
├── data/
│   ├── evaluation/
│   │   ├── images/
│   │   ├── vqa_evaluation_60.csv
│   │   └── vqa_evaluation_60.json
│   ├── sample_images/
│   ├── README_data.md
│   ├── sample_questions.csv
│   └── sample_vqa_pairs.csv
├── models/
├── notebooks/
├── outputs/
├── scripts/
│   ├── generate_synthetic_evaluation_set.py
│   └── ...
├── space/
│   ├── evaluation/
│   ├── samples/
│   ├── src/
│   ├── README.md
│   └── index.html
├── src/vqa/
├── tests/
├── DATASET_CARD.md
├── MODEL_CARD.md
├── README_HUGGINGFACE.md
├── requirements-ci.txt
└── requirements.txt
```

## Run the Static Space locally

```cmd
cd /d "C:\Users\atripathi\OneDrive - Veralto\Desktop\AI Codes\GIT Projects\transformer-projects\06-multimodal-visual-question-answering-transformer\space"
python -m http.server 8016
```

Open `http://localhost:8016/` in Chrome or Edge and perform a hard refresh with
`Ctrl + Shift + R` after replacing JavaScript files.

## Validate the committed evaluation suite

```bash
python scripts/generate_synthetic_evaluation_set.py --check
pytest
```

To regenerate the safe synthetic images and 60-pair records:

```bash
python scripts/generate_synthetic_evaluation_set.py
```

## Local Python reference setup

```bash
python -m venv .venv
pip install -r requirements.txt
python scripts/prepare_sample_data.py
python scripts/run_local_vilt.py data/sample_images/shapes_scene.png "What color is the circle?"
python scripts/evaluate_vqa.py --limit 3
python scripts/benchmark_latency.py --repeats 5
```

The committed output JSON files intentionally remain `not_evaluated`. Replace
them only with results generated on a documented dataset, browser, device, and
software configuration.

## Hugging Face Static Space deployment

1. Create a public Space named `06-multimodal-visual-question-answering-transformer`.
2. Select **Static** and the **Blank** template.
3. Upload the **contents** of `space/` to the root of the Space, not the outer
   Project 06 folder.
4. Confirm that the root Space README contains `sdk: static` and
   `app_file: index.html`.
5. Wait for the Space to rebuild after the commit.
6. Test an interactive answer and the evaluation lab in Chrome or Edge.
7. Add the live Space page link to this README and your portfolio.

For automatic GitHub-to-Hugging-Face synchronization, configure:

- GitHub Actions secret: `HF_TOKEN` with write access;
- GitHub Actions variable: `HF_SPACE_REPO` with value
  `anmol-unitmole/06-multimodal-visual-question-answering-transformer`.

## Responsible use and privacy

This project is for education and portfolio demonstration only. The model can
produce incomplete, incorrect, biased, or misleading answers. Do not use it for
medical, legal, financial, security, surveillance, identity verification,
employment, insurance, or other high-stakes decisions.

Do not upload private photographs, IDs, medical images, confidential workplace
images, proprietary documents, or copyrighted images without permission.

## Portfolio description

**One line:** Browser-deployed multimodal VQA system using SmolVLM,
Transformers.js, WebGPU, generation-confidence diagnostics, a balanced 60-pair
evaluation lab, category analysis, latency benchmarking, and failure analysis.

**Skills demonstrated:** multimodal AI, vision-language Transformers, VQA,
image preprocessing, question preprocessing, browser ONNX inference,
Transformers.js, WebGPU, confidence interpretation, evaluation design,
category-wise analysis, latency analysis, responsible AI, testing, CI, and
Hugging Face Static Spaces.
