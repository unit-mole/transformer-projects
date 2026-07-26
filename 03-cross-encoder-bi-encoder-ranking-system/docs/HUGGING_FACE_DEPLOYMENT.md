# Hugging Face Spaces Deployment Guide


## Current hosting eligibility note

Hugging Face currently treats Static Spaces as free for everyone. Gradio and
Docker Spaces use compute and generally require an eligible paid plan to create;
free personal accounts in good standing may host a limited number of Gradio
Spaces using the platform's ZeroGPU exception. CPU Basic has no hourly hardware
charge, but account eligibility for creating a compute-backed Space still
applies.

This repository remains Gradio-ready as requested. Check the current Spaces
Overview before deployment. When a compute-backed Space is unavailable, keep
the GitHub project complete and use a Static Space or browser-inference version
as a separate deployment fallback.

## Option A — Manual upload

1. Sign in to Hugging Face.
2. Create a new Space.
3. Choose a public Space and select **Gradio** as the SDK.
4. Use a name such as `cross-encoder-bi-encoder-ranking`.
5. Copy the contents of `03-cross-encoder-bi-encoder-ranking-system/` to the
   root of the Space repository.
6. Confirm these root files exist:
   - `README.md`
   - `app.py`
   - `gradio_app.py`
   - `requirements.txt`
   - `config.yaml`
   - `src/`
   - `data/`
7. Commit the files.
8. Wait for dependency installation and model download.
9. Run a sample query.
10. Copy the final Space URL into both GitHub README files.

The models are downloaded from the Hub. The Space does not train a model. When a
saved compatible index is absent, it creates only the 24-document NumPy sample
index.

## Option B — GitHub Actions sync

The included workflow can sync only this project subdirectory to a Space.

Configure:

- GitHub Actions secret: `HF_TOKEN`
- GitHub Actions repository variable: `HF_SPACE_ID`
  - Example: `your-username/cross-encoder-bi-encoder-ranking`

Push to `main`. The CI runs first; the sync job runs only when `HF_SPACE_ID` is
configured.

Use a fine-grained Hugging Face token with write permission only to the target
Space.

## Large files

Do not commit large model weights. Load pretrained models from the Hub.

For a larger saved index:

- track large files with Git LFS;
- keep document metadata aligned with index IDs;
- document the source dataset and license;
- never upload private or proprietary documents;
- consider a dedicated dataset or model repository for versioned artifacts.

## Troubleshooting

### Space is slow on first request

The first request may download and initialize both models. Subsequent requests
reuse the process cache.

### Out-of-memory condition

Reduce candidate K, use a smaller reranker such as
`cross-encoder/ms-marco-TinyBERT-L-2-v2`, or prebuild the index.

### Model download failure

Check Space logs, verify model IDs, and confirm outbound network availability.

### Gradio build failure

Confirm that `app.py` is at the root and imports `build_demo` from
`gradio_app.py`. Avoid moving `src/`, `data/`, or `config.yaml`.
