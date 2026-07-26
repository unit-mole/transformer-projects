# Hugging Face Spaces Deployment Guide

## 1. Create the Space

1. Sign in to Hugging Face.
2. Open **Spaces** and choose **Create new Space**.
3. Name it, for example, `english-hindi-neural-machine-translation`.
4. Select **Gradio** as the SDK.
5. Choose a public visibility for a portfolio demo.
6. Select an available CPU or ZeroGPU-compatible option based on your account.

## 2. Copy files

The Space repository root must contain:

```text
app.py
gradio_app.py
requirements.txt
README.md
MODEL_CARD.md
configs/
data/
models/
src/
```

Copy the contents of `02-neural-machine-translation-transformer/` into the Space root. Do not nest the whole `transformer-projects` repository inside the Space.

## 3. Space metadata

Keep the YAML block at the top of `README.md`:

```yaml
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
```

## 4. Dependencies

`requirements.txt` installs PyTorch, Transformers, SentencePiece, Gradio, pandas, SacreBLEU, and supporting packages.

## 5. Model loading

By default:

```text
Helsinki-NLP/opus-mt-en-hi
Helsinki-NLP/opus-mt-hi-en
```

The Space downloads a direction only when that direction is first used. No training occurs at startup.

To use your fine-tuned models, set Space variables:

```text
EN_HI_MODEL_ID=<username>/<en-hi-model-repo>
HI_EN_MODEL_ID=<username>/<hi-en-model-repo>
```

## 6. Large files

Preferred order:

1. store checkpoints in a Hugging Face model repository;
2. load them by model ID;
3. use Git LFS only when repository-local artifacts are truly necessary;
4. never commit `.bin` or `.safetensors` files as ordinary Git blobs.

## 7. Test the Space

Test:

- English→Hindi automatic mode;
- Hindi→English automatic mode;
- manual direction;
- mixed-language handling;
- CSV upload;
- output CSV download;
- model details and responsible-use text.

## 8. Add links to GitHub

Replace:

```text
<YOUR_HUGGINGFACE_SPACE_URL>
<YOUR_HUGGINGFACE_MODEL_URL>
<YOUR_GITHUB_REPOSITORY_URL>
```

in the project and root READMEs.

## Current plan note

Hugging Face’s current Spaces documentation states that Static Spaces are free for everyone, while Gradio/Docker creation uses hosted compute and may require a paid plan; eligible free personal accounts can host a limited number of ZeroGPU Gradio Spaces. Confirm the options shown for your account when creating the Space.
