# Project 01 RTX Fine-Tuning and Evaluation Guide

The complete notebook performs CUDA validation, deterministic CNN/DailyMail subset creation, Lead-3/TextRank baselines, pretrained DistilBART evaluation, mixed-precision fine-tuning, fine-tuned evaluation, ROUGE, BERTScore, latency, compression, numeric-risk analysis, optional real LSTM comparison, error analysis, charts, JSON/CSV/Markdown promotion, and optional Hugging Face Model Hub upload.

## Recommended run

The default `portfolio` profile uses 5,000 train, 500 validation, and 500 held-out test examples for two epochs. It is a reproducible portfolio benchmark, not a full-dataset state-of-the-art claim.

```bat
cd /d "C:\Users\atripathi\OneDrive - Veralto\Desktop\AI Codes\GIT Projects\transformer-projects\01-abstractive-text-summarization-transformer"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
python scripts\check_gpu.py
jupyter lab
```

Open `notebooks/complete_distilbart_training_evaluation_pipeline.ipynb` and run every cell in order.

## LSTM comparison

The notebook writes `data/benchmark_cache/lstm_benchmark_input.csv`. Run the previous LSTM model on those IDs and save real predictions to `data/lstm_predictions.csv` with columns `id,lstm_summary,lstm_inference_seconds`. Rerun the comparison cells. No LSTM metric is invented.

## Git policy

Commit code, notebook, configuration, JSON/CSV/Markdown/PNG results, and reviewed examples. Do not commit downloaded dataset caches, virtual environments, trainer checkpoints, or model weight files. Publish the fine-tuned checkpoint to a Hugging Face model repository.
