# Hugging Face Spaces Deployment Guide

## Important account eligibility note

As of July 2026, Hugging Face documentation states that Static Spaces are free
for everyone, while creating Gradio or Docker Spaces that use hosted compute
generally requires a paid account. Free personal accounts in good standing may
host up to two Gradio Spaces using ZeroGPU. Check the current Spaces page before
deployment because eligibility can change.

This repository is fully **Gradio Space-ready**. For a strictly free portfolio,
use this project as one of the eligible ZeroGPU Gradio Spaces, when available,
or run it locally until compute access is available.

## Option A — Create the Space in the web interface

1. Sign in to Hugging Face.
2. Select **New Space**.
3. Name it `long-document-qa-longformer`.
4. Choose **Gradio** as the SDK.
5. Choose public visibility for a portfolio demo.
6. Select eligible ZeroGPU or available CPU hardware.
7. Copy the contents of
   `04-long-document-question-answering-longformer/` to the Space repository
   root. `app.py`, `requirements.txt`, and `README.md` must be at the root.
8. Replace `<YOUR_USERNAME>` placeholders.
9. Commit the files. The Space will install dependencies and start `app.py`.
10. Test a sample document before sharing the URL.

## Option B — Push with Git

```bash
git lfs install
git clone https://huggingface.co/spaces/<YOUR_USERNAME>/long-document-qa-longformer
cd long-document-qa-longformer

# Copy project files into this folder, then:
git add .
git commit -m "Deploy Longformer long-document QA demo"
git push
```

## Required files

- `app.py`
- `gradio_app.py`
- `requirements.txt`
- `README.md` with Space YAML metadata
- `src/`
- `data/sample_documents/`
- `data/sample_questions.csv`
- `models/model_metadata.json`
- `MODEL_CARD.md`

## Model loading

The app downloads
`valhalla/longformer-base-4096-finetuned-squadv1` from the Hugging Face Hub on
the first build. No training occurs at startup.

For a custom fine-tuned model, upload weights to a separate model repository and
set a Space variable:

```text
LONGDOCQA_MODEL_ID=<YOUR_USERNAME>/<YOUR_MODEL_REPOSITORY>
```

Large weights should use Hugging Face Hub storage and Git LFS rather than normal
GitHub commits.

## Recommended Space variables

```text
LONGDOCQA_MAX_LENGTH=2048
LONGDOCQA_STRIDE=256
LONGDOCQA_MAX_UPLOAD_MB=10
LONGDOCQA_MAX_DOCUMENT_CHARACTERS=2000000
LONGDOCQA_DEVICE=auto
```

## Validation checklist

- sample selector loads all three documents
- TXT and Markdown uploads work
- CSV text extraction works
- selectable-text PDF extraction works
- scanned PDF returns a clear no-OCR message
- answer and evidence refer to the same span
- confidence is labelled as a proxy
- long documents report more than one token window
- private-data warning is visible
- GitHub and model links are updated

## Shareable URL

`https://huggingface.co/spaces/<YOUR_USERNAME>/long-document-qa-longformer`

Add that URL to the project README and the main portfolio README.
