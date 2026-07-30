# Hugging Face Spaces Deployment Guide

Deploy only after the full notebook has produced reviewed adapter and evaluation artifacts.

## 1. Train and review the adapter

Use:

```text
notebooks/05_full_training_evaluation_pipeline.ipynb
```

or the command-line stages:

```bash
python scripts/generate_enhanced_dataset.py --target-examples 600
python scripts/train_lora.py --base-model google/flan-t5-base --epochs 6
python scripts/evaluate_base_vs_lora.py
python scripts/check_portfolio_readiness.py
```

Complete the dataset and response human-review gates before promotion.

## 2. Publish the adapter to Hugging Face Hub

Create a model repository such as:

```text
<your-huggingface-username>/flan-t5-base-ml-ds-lora
```

Upload the contents of the reviewed `lora_adapter` directory, including:

```text
adapter_config.json
adapter_model.safetensors
experiment_metadata.json
README.md or MODEL_CARD.md
```

The model card must identify `google/flan-t5-base` as the base model and report only real evaluation values.

Example CLI after authentication:

```bash
hf auth login
hf upload <your-huggingface-username>/flan-t5-base-ml-ds-lora \
  outputs/experiments/<run>/training/lora_adapter .
```

## 3. Create the Gradio Space

1. Select **New Space**.
2. Name it `ml-ds-instruction-tuned-assistant` or another professional name.
3. Choose the **Gradio** SDK.
4. Start with CPU Basic and verify latency; upgrade hardware only if necessary.
5. Use a compatible project license and preserve the base-model license notice.

## 4. Copy deployment files

The Space root should contain:

- `app.py`
- `gradio_app.py`
- `requirements.txt`
- `README.md` with Space YAML metadata
- `src/`
- `data/sample_instructions.jsonl`
- promoted evaluation JSON files for display
- model and dataset cards

Do not upload raw teacher generations, checkpoints, local caches, or the complete training environment.

## 5. Configure Space variables

```text
ADAPTER_ID=<your-huggingface-username>/flan-t5-base-ml-ds-lora
BASE_MODEL_ID=google/flan-t5-base
MAX_INPUT_LENGTH=512
MAX_TARGET_LENGTH=256
MERGE_ADAPTER=false
```

A public adapter does not require a token. Use a Space secret only for private repositories.

## 6. Verify startup behavior

Training must never start in the Space. The first request loads the base model and adapter.

Check inference metadata:

- `model_mode` is `lora_adapter`;
- `adapter` shows the correct Hub repository;
- `base_model` is `google/flan-t5-base`;
- `base_model_fallback` is not displayed.

## 7. Validate the deployed artifact

Ask prompts from several categories and new prompts outside the benchmark. Compare representative outputs with local results. Confirm that the Space uses the same adapter revision that was evaluated.

## 8. Publish links and evidence

Update GitHub and the Hugging Face model card with:

- Space URL;
- adapter URL;
- repository URL;
- dataset and benchmark description;
- real base-versus-LoRA metrics;
- limitations and responsible-use notes;
- screenshots of the app and evaluation artifacts.

## Large Files

Keep base-model weights, checkpoints, caches, and raw experiment folders out of ordinary Git history. Publish the compact adapter to Hugging Face Hub. Commit only reviewed JSON/CSV/PNG/Markdown evidence to GitHub.
