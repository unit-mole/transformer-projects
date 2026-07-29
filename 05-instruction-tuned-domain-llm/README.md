---
title: ML Data Science Instruction Tuned Assistant
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.13.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
suggested_hardware: cpu-basic
---

# 05 — Instruction-Tuned Domain LLM

[![CI](https://github.com/YOUR_GITHUB_USERNAME/transformer-projects/actions/workflows/05-instruction-tuned-domain-llm.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/transformer-projects/actions/workflows/05-instruction-tuned-domain-llm.yml)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/YOUR_HF_USERNAME/ml-ds-instruction-tuned-assistant)

> **One-line portfolio description:** A FLAN-T5 ML/Data Science Learning Assistant adapted with LoRA / PEFT, evaluated for instruction adherence, semantic similarity, relevance, latency, and hallucination risk, and deployed with Gradio on Hugging Face Spaces.

## Responsible use

This project is for educational and portfolio demonstration purposes. The assistant may generate incomplete, incorrect, outdated, biased, or hallucinated responses. It is designed for ML and Data Science learning support—not legal, medical, financial, immigration, safety-critical, official, or autonomous decision-making. Do not paste private, confidential, proprietary, copyrighted, or personally identifiable information into the public demo. Human review is required before using any generated explanation, code, or recommendation.

## Project pattern

| Item | Selection |
|---|---|
| Project | `05-instruction-tuned-domain-llm` |
| Application | Fine-tune a small instruction-following model using LoRA / PEFT |
| Assistant theme | ML and Data Science Learning Assistant |
| Base model | `google/flan-t5-small` |
| Dataset | 401-example public-safe ML/DS instruction curriculum with topic-grouped splits |
| Evaluation | Instruction adherence, BERTScore, response relevance, hallucination analysis, manual review, latency |
| Deployment | Hugging Face Spaces with Gradio |

## Why this project matters

Instruction tuning teaches a pretrained model to respond to natural-language tasks rather than merely continue text. The assistant supports concept explanations, algorithm comparisons, metric explanations, examples, interview answers, project workflows, and quality analytics learning scenarios. It demonstrates the progression from classical ML and sequence models toward parameter-efficient Generative AI systems.

## Architecture

![Architecture](images/architecture.png)

```text
Instruction dataset → validation and prompt formatting → FLAN-T5-small
→ LoRA / PEFT adapter training → held-out evaluation → Gradio inference app
```

## Dataset

The repository contains an original 82-example seed curriculum and an expanded **401-example public-safe instruction dataset** across **9 capability categories**. The extended dataset uses **203 topic groups** so paraphrases of the same concept remain in one split and cannot inflate held-out results. No private company data is included.

| Statistic | Value |
|---|---:|
| Extended examples | 401 |
| Train / validation / test | 323 / 42 / 36 |
| Topic groups | 203 |
| Average prompt words | 7.79 |
| Average response words | 43.84 |

See [DATASET_CARD.md](DATASET_CARD.md) and [data/README_data.md](data/README_data.md).

## Prompt template

```text
Context: You are an educational ML and Data Science learning assistant...

Instruction:
{instruction}

Input:
{optional_input}

Response:
```

The same formatter is used during training and inference.

## LoRA / PEFT design

LoRA keeps the pretrained FLAN-T5 weights frozen and trains small low-rank matrices in selected attention modules. This project uses `SEQ_2_SEQ_LM` and targets the T5 `q` and `v` modules. The original lightweight profile uses rank 8 and alpha 16; the portfolio-scale RTX experiment uses rank 16, alpha 32, and dropout 0.05. The resulting adapter is smaller and easier to publish than a fully fine-tuned model.

## Training workflow

```bash
python scripts/build_extended_dataset.py
python scripts/run_complete_experiment.py
```

For the recommended step-by-step RTX workflow, run `notebooks/05_end_to_end_gpu_lora_training_evaluation.ipynb`. Training happens locally on the GPU; the public Space performs inference only.

## Evaluation

```bash
# Recommended complete GPU experiment
python scripts/run_complete_experiment.py

# Or use the step-by-step notebook
jupyter lab notebooks/05_end_to_end_gpu_lora_training_evaluation.ipynb
```

The repository intentionally ships with `status: not_run` until the RTX notebook is executed. The notebook compares the untouched base model and LoRA adapter on the same 36 topic-isolated held-out prompts and writes auditable per-example outputs.

| Metric | Purpose | Important limitation |
|---|---|---|
| Held-out loss and perplexity | Measures target-token prediction on unseen prompts | Not a complete measure of answer usefulness |
| Instruction adherence | Checks category-specific task and format behavior | Transparent heuristic rubric requires human review |
| BERTScore precision/recall/F1 | Measures semantic similarity to reference answers | Not a factuality measure |
| ROUGE-1/2/L | Measures lexical overlap with references | Penalizes valid paraphrases |
| Semantic relevance | Sentence-Transformer similarity to prompt and reference | Embedding similarity is a proxy |
| Reference-support and risk flags | Flags low support, unsupported numbers, attributions, and absolutes | Triage only; not proof of hallucination |
| Latency, throughput, and GPU memory | Measures deployment behavior on the recorded hardware | Hardware and decoding dependent |
| Paired bootstrap intervals | Quantifies uncertainty in base-versus-LoRA deltas | Applies only to this held-out set |
| Manual review | Scores correctness, relevance, clarity, preference, and hallucinations | Reviewer judgment must be documented |


## Portfolio-scale executed experiment

Project 05 now includes a complete RTX experiment notebook:

```text
notebooks/05_end_to_end_gpu_lora_training_evaluation.ipynb
```

It expands the curriculum, validates topic-group isolation, trains a real LoRA adapter, evaluates base FLAN-T5 and the adapter with identical deterministic decoding, produces paired confidence intervals, saves predictions and charts, and creates a manual factuality-review template. See [PORTFOLIO_EXPERIMENT_GUIDE.md](PORTFOLIO_EXPERIMENT_GUIDE.md) and [MANUAL_EVALUATION_RUBRIC.md](MANUAL_EVALUATION_RUBRIC.md).

Expected result artifacts include:

```text
outputs/portfolio_experiment/model_metrics.json
outputs/portfolio_experiment/base_model_metrics.json
outputs/portfolio_experiment/lora_model_metrics.json
outputs/portfolio_experiment/base_vs_lora_comparison.json
outputs/portfolio_experiment/base_vs_lora_per_example.csv
outputs/portfolio_experiment/category_metrics.csv
outputs/portfolio_experiment/manual_review_results.csv
outputs/portfolio_experiment/hallucination_analysis.md
outputs/portfolio_experiment/before_after_finetuning_examples.md
outputs/portfolio_experiment/*.png
```

Numeric results must be copied into the public README and model card only after the notebook completes and the manual review is performed.

## Gradio application

The app provides categories, sample prompts, optional context, generation controls, model details, current metric status, limitations, responsible-use guidance, and a base-vs-adapter selection. Model loading is lazy so CI can import the app without downloading weights.

## Run locally

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/transformer-projects.git
cd transformer-projects/05-instruction-tuned-domain-llm
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python scripts/build_extended_dataset.py
python app.py
```

For GPU training and the complete evaluation notebook, install `requirements-training.txt`.

To use the trained adapter:

```bash
# Windows PowerShell
$env:ADAPTER_MODEL_ID="YOUR_HF_USERNAME/ml-ds-instruction-tuned-flan-t5-small-lora"

# macOS/Linux
export ADAPTER_MODEL_ID="YOUR_HF_USERNAME/ml-ds-instruction-tuned-flan-t5-small-lora"
```

## Recommended three-part portfolio deployment

| Component | Role |
|---|---|
| GitHub repository | Complete Python training, LoRA / PEFT, evaluation, tests, Gradio, and JavaScript deployment code |
| Hugging Face Model Hub | LoRA adapter repository plus a merged, ONNX browser-model repository after real training |
| Hugging Face Static Space | Live Transformers.js assistant performing inference directly in the visitor’s browser |

The existing Gradio application remains part of the engineering project. For a strictly static public demo, deploy the contents of [`web/`](web/) as a separate Static Space. The default browser demo uses `Xenova/flan-t5-small` and identifies it as a base-model demonstration. After training, merge the adapter, export the merged checkpoint to ONNX, and enter your own Hub model ID in the interface.

See [STATIC_BROWSER_DEPLOYMENT_ROADMAP.md](STATIC_BROWSER_DEPLOYMENT_ROADMAP.md) and [DEPLOYMENT_STATIC_SPACE.md](DEPLOYMENT_STATIC_SPACE.md).

## Hugging Face Spaces deployment

Create a Gradio Space, copy this project’s contents to the Space root, set `BASE_MODEL_ID` and `ADAPTER_MODEL_ID`, and push. See [DEPLOYMENT_HUGGINGFACE.md](DEPLOYMENT_HUGGINGFACE.md). The final public URL will be:

```text
https://huggingface.co/spaces/YOUR_HF_USERNAME/ml-ds-instruction-tuned-assistant
```

## Project structure

```text
05-instruction-tuned-domain-llm/
├── app.py
├── gradio_app.py
├── configs/
├── data/
├── notebooks/
├── src/
├── scripts/
├── tests/
├── web/
├── models/
├── outputs/
├── images/
├── MODEL_CARD.md
├── DATASET_CARD.md
├── PORTFOLIO_EXPERIMENT_GUIDE.md
├── MANUAL_EVALUATION_RUBRIC.md
├── requirements-training.txt
├── DEPLOYMENT_HUGGINGFACE.md
├── DEPLOYMENT_STATIC_SPACE.md
├── STATIC_BROWSER_DEPLOYMENT_ROADMAP.md
├── VALIDATION_REPORT.md
├── requirements-export.txt
└── requirements.txt
```

## Screenshots to add after deployment

Capture the Space landing page, a concept explanation, an algorithm comparison, a code example, the model metadata tab, the evaluation tab after real metrics exist, and the mobile layout. Save them in `images/` and replace the placeholder section only after the public app is working.

## Skills demonstrated

Transformer architecture, FLAN-T5, instruction tuning, LoRA, PEFT, custom dataset design, prompt formatting, sequence-to-sequence training, model cards, dataset cards, responsible AI, BERTScore, evaluation design, hallucination analysis, Gradio, Hugging Face Spaces, testing, CI, and deployment documentation.

## Career positioning

For a Quality Data Scientist moving toward ML, Applied AI, and Generative AI roles, this project shows how to convert domain learning needs into an end-to-end assistant: safe data design, parameter-efficient adaptation, measurable evaluation, transparent limitations, and a public deployment. The quality analytics examples connect the system naturally to case prioritization, defect trends, root-cause classification, technical onboarding, and future internal knowledge assistants.

## Future improvements

Add a second human reviewer, increase the number of independently reviewed examples, execute generated code in a sandbox, compare FLAN-T5-small with a compact causal instruction model, add retrieval grounding with citations, test multilingual prompts, and publish the merged quantized ONNX model after verifying browser quality.
