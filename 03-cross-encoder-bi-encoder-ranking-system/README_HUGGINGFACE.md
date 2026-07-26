# Hugging Face Portfolio Structure

Project 03 uses three Hugging Face-facing components.

## 1. Static Space

Source:

```text
web/
```

Build:

```bash
cd web
npm install
npm run build
```

Deploy:

```bash
python scripts/deploy_static_space.py
```

The Space runs real browser-based MiniLM retrieval and MS MARCO reranking.

## 2. Pipeline model card repository

Source:

```text
model_hub/pipeline-card/
```

Publish:

```bash
python scripts/publish_pipeline_card.py
```

This repository documents the complete system and credits the original model
owners. It does not claim that pretrained weights were newly trained.

## 3. Future fine-tuned model repositories

Templates:

```text
model_hub/bi-encoder-template/
model_hub/cross-encoder-template/
```

Only publish those as personal model repositories after genuine fine-tuning or
your own validated conversion.
