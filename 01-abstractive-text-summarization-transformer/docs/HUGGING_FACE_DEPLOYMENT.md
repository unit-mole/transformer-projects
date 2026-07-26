# Hugging Face Deployment Guide

## Current Eligibility Note

As of July 2026, Hugging Face documentation states that creating a new Gradio or Docker Space requires an eligible paid plan, while Static Spaces remain free. CPU Basic is listed at no hourly hardware charge but is still compute-backed. Check the latest Space creation rules before deployment.

## Gradio Space Deployment

1. Sign in to Hugging Face.
2. Create a new Space.
3. Choose **Gradio** as the SDK.
4. Choose a public Space for a portfolio demo.
5. Use CPU Basic unless you intentionally select paid hardware.
6. Copy the contents of `01-abstractive-text-summarization-transformer/` to the Space repository root.
7. Confirm the Space root contains:
   - `README.md` with YAML metadata;
   - `app.py`;
   - `gradio_app.py`;
   - `requirements.txt`;
   - `src/`;
   - `data/sample_articles.csv`;
   - `MODEL_CARD.md`.
8. Push the files using the web interface or Git.
9. Review build logs. The model is downloaded on the first inference request; it is not trained.
10. Replace URL placeholders in the GitHub READMEs after the Space is live.

## Git Commands

```bash
git clone https://huggingface.co/spaces/<YOUR_USERNAME>/abstractive-text-summarization-transformer
cd abstractive-text-summarization-transformer
# Copy project files here
git add .
git commit -m "Deploy DistilBART summarization Gradio app"
git push
```

## Large Fine-Tuned Weights

Prefer a dedicated Hugging Face model repository:

```bash
huggingface-cli login
git lfs install
```

Upload `models/transformer_summarization_model/` and `models/tokenizer/`, then set `MODEL_NAME=<YOUR_USERNAME>/<MODEL_REPO>` in the Space variables. Do not bundle multi-gigabyte PyTorch files in the GitHub portfolio repository.

## Strictly Free Alternative

A Static Space requires a browser inference implementation, normally Transformers.js/ONNX. That is a different runtime architecture from this Python/Gradio project. The present repository keeps the requested Gradio implementation honest and deployment-ready; create a static variant only when free-account eligibility requires it.
