# Hugging Face Spaces Deployment Guide

## 1. Train and upload the adapter

Run training in Colab, Kaggle, or a local GPU environment:

```bash
python scripts/prepare_dataset.py
python scripts/train_lora.py
```

Create a Hugging Face **model** repository and upload the contents of `models/lora_adapter/`. The adapter is much smaller than a merged full model. Git LFS is recommended for large binary artifacts.

## 2. Create the Space

1. Create a new Hugging Face Space.
2. Select **Gradio** as the SDK.
3. Choose CPU Basic for the lightweight FLAN-T5-small demo.
4. Copy the project files to the Space repository root.
5. Replace the Space `README.md` with `README_HUGGINGFACE.md` or copy its YAML metadata to the top of the main README.

## 3. Configure variables

Set these Space variables:

```text
BASE_MODEL_ID=google/flan-t5-small
ADAPTER_MODEL_ID=YOUR_HF_USERNAME/ml-ds-instruction-tuned-flan-t5-small-lora
```

Use a Space secret only if the adapter repository is private. A public portfolio adapter does not require a token.

## 4. Required root files

- `app.py`
- `gradio_app.py`
- `requirements.txt`
- `README.md` with Space YAML metadata
- `src/`
- `outputs/model_metrics.json`
- `models/model_metadata.json`

The Space installs Python packages from `requirements.txt` and runs `app.py`. It loads model artifacts for inference and never starts training.

## 5. Test before sharing

Verify representative prompts from every category, adapter and base-model modes, first-request latency, error handling, disclaimers, and that no invented evaluation numbers appear. Add the final Space URL to the GitHub README only after the app builds successfully.
