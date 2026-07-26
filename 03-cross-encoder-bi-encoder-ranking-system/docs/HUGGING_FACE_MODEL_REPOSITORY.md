# Hugging Face Model Repository Guide

## Publish the pipeline card

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Set the token:

Windows PowerShell:

```powershell
$env:HF_TOKEN="<YOUR_TOKEN>"
```

macOS/Linux:

```bash
export HF_TOKEN="<YOUR_TOKEN>"
```

Publish:

```bash
python scripts/publish_pipeline_card.py
```

Default repository:

```text
anmol-unitmole/docrank360-ranking-pipeline-card
```

Override it with:

```bash
python scripts/publish_pipeline_card.py \
  --repo-id anmol-unitmole/<YOUR_REPOSITORY_NAME>
```

## What is uploaded

```text
model_hub/pipeline-card/
├── README.md
├── pipeline_config.json
└── evaluation_results.json
```

No pretrained weights are copied.

## Before publishing metrics

Run:

```bash
python scripts/evaluate_model.py
python scripts/benchmark_latency.py
```

Then transfer only the measured values into:

```text
model_hub/pipeline-card/evaluation_results.json
```

Do not publish placeholder values as completed results.
