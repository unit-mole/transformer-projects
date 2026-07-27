# 09 — Vision-Language Image-Text Retrieval with CLIP

[![Deployment](https://img.shields.io/badge/Deployment-GitHub%20Pages-222?logo=github)](https://unit-mole.github.io/transformer-projects/09-vision-language-image-text-retrieval-clip/)
[![Browser AI](https://img.shields.io/badge/Inference-In%20Browser-6b5cff)](#browser-architecture)
[![Model](https://img.shields.io/badge/Model-CLIP%20ViT--B%2F32-orange)](MODEL_CARD.md)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](../../actions)

A browser-based multimodal AI application for natural-language image retrieval and zero-shot image classification. The app is designed specifically for static GitHub Pages hosting: there is no Python backend, server-side inference API, database, Streamlit, Gradio, Flask, or FastAPI.

## Responsible-use and image-safety notice

This project is for educational and portfolio demonstration purposes only. CLIP results may be incomplete, biased, irrelevant, or misleading. Similarity scores are model alignment scores, not calibrated probabilities. Zero-shot predictions can be wrong for ambiguous, low-quality, out-of-distribution, or sensitive images.

Do not use this project for medical, legal, financial, safety-critical, surveillance, identity verification, security, hiring, insurance, quality-release, or official decision-making. Do not upload private photos, IDs, medical images, confidential workplace images, proprietary material, or copyrighted images without permission. The public gallery uses original synthetic demo images only.

## Strict project pattern

| Field | Value |
|---|---|
| Project number | 09 |
| Project name | `09-vision-language-image-text-retrieval-clip` |
| Application | CLIP-based image-text retrieval application |
| User workflow | Enter a natural-language query and retrieve matching gallery images |
| Additional feature | Zero-shot image classification for an uploaded image |
| Model | CLIP ViT-B/32 using quantized ONNX weights through Transformers.js |
| Dataset | Small original synthetic gallery; scripts support a Flickr8k subset without redistributing it |
| Metrics | Recall@1, Recall@5, Recall@10, similarity-score analysis, and latency |
| Deployment | GitHub Pages |

## Live demo

**GitHub Pages URL:** `https://unit-mole.github.io/transformer-projects/09-vision-language-image-text-retrieval-clip/`

The first CLIP request downloads quantized browser model assets and may take longer. Assets are cached by the browser. A clearly labeled caption-search baseline is used only when the CLIP model cannot load.

## What the project demonstrates

- CLIP vision-language modeling and a shared image-text embedding space
- Natural-language image search with browser-side cosine similarity
- Zero-shot image classification with editable candidate labels
- Quantized ONNX inference through Transformers.js and ONNX Runtime Web
- Static GitHub Pages deployment with no backend
- Offline gallery preparation, embedding generation, Recall@K evaluation, similarity analysis, and latency benchmarking
- Professional model, dataset, testing, CI, and responsible-AI documentation

## Browser architecture

The deployed application uses a deployment-safe hybrid approach:

1. Gallery metadata and safe images are served as static files.
2. The app first looks for validated precomputed CLIP image embeddings in `web/data/image_embeddings.json`.
3. When embeddings are not bundled, the browser loads the quantized CLIP vision encoder, creates gallery embeddings once, and caches them locally.
4. A user query is encoded by the quantized CLIP text encoder in the browser.
5. Cosine similarity ranks the gallery images.
6. For zero-shot classification, the browser embeds the uploaded image and text prompts such as `a photo of a {label}` and ranks the labels.
7. If CLIP assets cannot load, the UI explicitly switches to a caption-based baseline rather than pretending that baseline output is CLIP.

Transformers.js executes compatible ONNX models in the browser through ONNX Runtime Web. No uploaded image is sent to an application backend.

## Why CLIP ViT-B/32

CLIP ViT-B/32 offers a strong portfolio balance: recognizable vision-language capability, 512-dimensional shared embeddings, zero-shot classification, broad tooling support, and quantized browser-compatible ONNX assets. It is still a substantial model, so the project uses a small gallery, browser caching, and sequential gallery embedding generation.

## Dataset

The committed public demo contains 12 original synthetic PNG scenes and structured metadata. This keeps the repository safe, reproducible, and deployable. The preparation scripts can be adapted to a Flickr8k subset, but the full Flickr8k dataset is intentionally not included.

See [`DATASET_CARD.md`](DATASET_CARD.md) and [`data/README_data.md`](data/README_data.md).

## CLIP in recruiter-friendly language

CLIP contains an image encoder and a text encoder. Both convert their inputs into vectors in the same semantic space. Images and text that describe similar content should have vectors that point in similar directions. This project uses cosine similarity to rank gallery images for a natural-language query. For zero-shot classification, it compares an uploaded image against text prompts representing candidate classes.

## Similarity-score interpretation

Higher cosine similarity generally means stronger image-text alignment. The scores are not calibrated probabilities and should not be treated as certainty. The zero-shot table also reports softmax-normalized values for easier comparison within the candidate set, but those values depend on the labels supplied by the user.

## Evaluation

The repository includes real evaluation code and an evaluation-query set. Metrics remain `null` until the scripts are executed with generated embeddings; no results are invented.

- **Recall@1:** relevant image appears as the first result.
- **Recall@5:** relevant image appears within the first five results.
- **Recall@10:** relevant image appears within the first ten results.
- **Similarity analysis:** score ranges, margins, ambiguous prompts, and low-confidence cases.
- **Latency:** text encoding, image encoding, similarity ranking, and total query time.
- **Baselines:** keyword and TF-IDF caption search can be compared with CLIP retrieval.

## Folder structure

```text
transformer-projects/
├── 09-vision-language-image-text-retrieval-clip/
│   ├── README.md
│   ├── README_GITHUB_PAGES.md
│   ├── MODEL_CARD.md
│   ├── DATASET_CARD.md
│   ├── requirements.txt
│   ├── requirements-model.txt
│   ├── package.json
│   ├── data/
│   ├── notebooks/
│   ├── src/
│   ├── scripts/
│   │   ├── sync_docs.py
│   │   ├── check_relative_paths.py
│   │   └── validate_web_assets.mjs
│   ├── tests/
│   ├── outputs/
│   ├── images/
│   └── web/                         # development app
│       ├── index.html
│       ├── style.css
│       ├── app.js
│       ├── data/
│       ├── sample_images/
│       └── model/
├── docs/
│   ├── .nojekyll
│   └── 09-vision-language-image-text-retrieval-clip/  # GitHub Pages copy
│       ├── index.html
│       ├── style.css
│       ├── app.js
│       ├── data/
│       ├── sample_images/
│       └── model/
└── .github/workflows/
    └── 09-vision-language-image-text-retrieval-clip.yml
```

## Local run: browser demo

```bash
cd 09-vision-language-image-text-retrieval-clip
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/run_local_web_server.py
```

Open `http://localhost:8000`. Do not open `web/index.html` directly with `file://`; browsers block JSON/model loading in that mode.

## Offline CLIP workflow

Install the optional model stack:

```bash
pip install -r requirements-model.txt
```

Then run:

```bash
python scripts/prepare_gallery.py
python scripts/generate_image_embeddings.py
python scripts/evaluate_retrieval.py
python scripts/benchmark_latency.py
python scripts/export_browser_assets.py
```

Generated browser embeddings are written to `web/data/image_embeddings.json`. Evaluation results are written to `outputs/`. After changing any browser asset, run `python scripts/sync_docs.py` so the deployment copy under `../docs/09-vision-language-image-text-retrieval-clip/` stays identical.

## ONNX export and quantization

The deployed default loads browser-ready quantized assets from the Hugging Face model repository. Custom export utilities are also included:

```bash
python scripts/export_clip_to_onnx.py --output-dir models/onnx_export
python scripts/quantize_onnx_model.py \
  --input models/onnx_export/model.onnx \
  --output models/onnx_export/model_quantized.onnx
```

Large ONNX files are intentionally excluded from Git. See `web/model/README.md` for optional self-hosting.

## GitHub Pages deployment

This repository already publishes from **`main /docs`**. Project 09 therefore keeps two synchronized copies of the static app:

- Development source: `09-vision-language-image-text-retrieval-clip/web/`
- Published copy: `docs/09-vision-language-image-text-retrieval-clip/`

After editing `web/`, run:

```bash
python scripts/sync_docs.py
python scripts/sync_docs.py --check
```

Then commit both folders and push to `main`. GitHub's built-in Pages build detects the `/docs` change automatically. The Project 09 workflow is validation-only and deliberately contains no `configure-pages`, `deploy-pages`, `gh-pages`, or token-based deployment steps.

See [`README_GITHUB_PAGES.md`](README_GITHUB_PAGES.md) for the exact deployment standard and troubleshooting.

## Screenshots to include

1. Home page with architecture and model-status card.
2. Retrieval results for `a red car on a road` with rank and cosine similarity.
3. Retrieval results for a more descriptive query such as `a quiet mountain lake under blue sky`.
4. Uploaded-image zero-shot classification with candidate labels.
5. Evaluation section after actual Recall@K metrics have been generated.
6. Validation workflow success and the built-in Pages deployment from `main /docs`.

## Portfolio positioning

**One-line description:** Browser-based CLIP image search and zero-shot image classification with quantized ONNX inference, Recall@K evaluation, and GitHub Pages deployment.

**Pinned repository description:** End-to-end multimodal AI portfolio project using CLIP ViT-B/32 for natural-language image retrieval and zero-shot classification directly in the browser.

This project connects naturally to Quality Data Science: the same architecture can support natural-language retrieval of inspection images, defect-example search, linking image evidence to quality notes, visual knowledge bases, and future multimodal RAG systems.

## Limitations

- Initial model download can be large and slower on mobile networks.
- Synthetic demo images are intentionally small and do not represent a production retrieval benchmark.
- CLIP can reflect biases from its pretraining data.
- Retrieval quality depends on gallery coverage and query wording.
- Softmax values are relative only to the supplied labels.
- Browser memory and execution-provider support vary by device.

## Future improvements

- Commit real precomputed embeddings after running the official generation script.
- Add a permitted Flickr8k evaluation subset without redistributing restricted images.
- Add WebGPU selection and IndexedDB model/embedding cache controls.
- Add image-to-image retrieval and multimodal query fusion.
- Add a lightweight visual defect-retrieval dataset relevant to quality analytics.
- Integrate the gallery with a future multimodal RAG assistant.
