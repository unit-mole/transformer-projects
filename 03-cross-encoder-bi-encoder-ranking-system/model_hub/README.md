# Hugging Face Model Hub Strategy

Project 03 uses pretrained base models and does not currently fine-tune or
convert new weights under the author's name.

## Recommended current repository

Create one transparent system-documentation repository:

```text
anmol-unitmole/docrank360-ranking-pipeline-card
```

It should contain:

- two-stage architecture;
- original base model links;
- browser-compatible ONNX model links;
- dataset description;
- preprocessing;
- evaluation status and metrics;
- latency results;
- limitations;
- responsible-use guidance;
- deployment links.

The ready-to-publish files are under:

```text
model_hub/pipeline-card/
```

Publish them with:

```bash
python scripts/publish_pipeline_card.py
```

## Do not claim the base models as newly trained models

The current implementation uses:

```text
sentence-transformers/all-MiniLM-L6-v2
cross-encoder/ms-marco-MiniLM-L-6-v2
Xenova/all-MiniLM-L6-v2
Xenova/ms-marco-MiniLM-L-6-v2
```

These should be credited as base or converted models.

## Future model repositories

Only create the following after genuine fine-tuning or your own validated
conversion:

```text
anmol-unitmole/docrank360-bi-encoder-retrieval
anmol-unitmole/docrank360-cross-encoder-reranker
```

Templates are included under:

```text
model_hub/bi-encoder-template/
model_hub/cross-encoder-template/
```

Do not publish those templates as trained-model repositories until the stated
training, conversion, and evaluation details are true.
