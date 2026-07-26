# 08 — Browser-Based Image Classification with a Vision Transformer

A static GitHub Pages application that runs compact Vision Transformer inference directly in the browser. The deployed starter uses a quantized ONNX conversion of `vit-tiny-patch16-224` through Transformers.js, reports top-k probabilities and client-side latency, and provides an honest perturbation-based patch-sensitivity explanation.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-222?logo=github)](https://unit-mole.github.io/transformer-projects/08-image-classification-vision-transformer/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions)](../../actions)
[![Browser AI](https://img.shields.io/badge/Inference-In%20Browser-6f42c1)](#browser-inference)

## Responsible use and image privacy

This project is for education and portfolio demonstration only. Predictions can be wrong, and confidence is only a model-produced score—not a guarantee. Do not use this application for medical, legal, financial, safety-critical, surveillance, identity-verification, security, hiring, insurance, product-release, or other official decisions. Do not upload private photos, IDs, confidential workplace images, copyrighted material without permission, or images containing personal or sensitive information. Human review is required for any real-world use.

## Strict project pattern

| Item | Implementation |
|---|---|
| Project | `08-image-classification-vision-transformer` |
| Application | Browser-based image classification using a compact Vision Transformer |
| Comparison | ViT versus a CNN/ResNet baseline framework |
| Outputs | Predictions, class probabilities, latency, evaluation artifacts, explainability visualization |
| Model | `onnx-community/vit-tiny-patch16-224-ONNX` for the directly deployable starter; scripts support fine-tuned ViT/DeiT export |
| Dataset | Starter demo: ImageNet-1k pretrained label space; training framework: CIFAR-10 by default |
| Metrics | Accuracy, macro F1, confusion matrix, parameter count, model size, and latency |
| Deployment | GitHub Pages—no Python backend or inference API |

## What is complete now

- Static HTML/CSS/JavaScript application that can be published directly to GitHub Pages.
- First-use model download from the Hugging Face Hub, followed by browser caching.
- Image upload, local preview, top-five predictions, probability bars, and measured client-side latency.
- Live occlusion/patch-sensitivity map that performs real repeated model inference.
- Python training, evaluation, attention-rollout, comparison, conversion, and benchmarking framework.
- Lightweight tests and GitHub Actions validation/deployment workflow.
- Model card, dataset card, deployment guide, metadata, output templates, and notebook starters.

## Important scope statement

No trained CIFAR-10 checkpoint, CNN/ResNet checkpoint, evaluation dataset, or historical CNN metrics were supplied with the request. Therefore, this repository **does not invent accuracy, macro F1, confusion-matrix, parameter, or comparative-latency results**. The included output files use `null`/`not_evaluated` values until the supplied scripts are run with real checkpoints and data.

The live starter model is a pretrained ImageNet classifier. Replace it with your fine-tuned CIFAR-10 or Intel-scene model before presenting task-specific claims.

## Why a compact Vision Transformer?

A Vision Transformer divides an image into patches, converts each patch to an embedding, and uses self-attention to learn relationships among patches. CNNs primarily learn local patterns through convolution filters; ViTs can model global patch relationships. A tiny ViT is suitable for this portfolio demo because its quantized browser model is much smaller than a base ViT and can execute on client devices without a backend.

## Browser inference

```text
User image
   ↓
Browser validates and decodes the file
   ↓
Transformers.js loads the model processor and quantized ONNX graph
   ↓
ONNX Runtime Web performs client-side inference
   ↓
Top-k labels, probabilities, and latency are rendered
```

The default model is downloaded from the Hugging Face Hub on first use and cached by the browser. No image is uploaded by this application. External model hosting is used only for public model assets. For a fully self-contained deployment, export/copy a compatible model into `web/model/` and update `web/metadata.json`.

## Explainability: attention versus patch sensitivity

The live web demo generates an **occlusion-based patch-sensitivity map** by masking image regions and measuring the reduction in the selected class score. This is a real perturbation experiment, but it is **not raw transformer attention**.

The Python module `src/attention_visualization.py` implements class-token attention rollout for models that return attention tensors. Run it with the final checkpoint to generate genuine precomputed attention examples, then place those assets in the web app. The repository deliberately avoids labeling an unrelated heatmap as attention.

## ViT versus CNN/ResNet comparison

Use the same dataset split, preprocessing intent, evaluation code, hardware, warm-up count, and number of benchmark runs. Save real results to `outputs/vit_vs_cnn_comparison.csv`.

| Model | Accuracy | Macro F1 | Parameters | Average latency | Model size | Status |
|---|---:|---:|---:|---:|---:|---|
| CNN / ResNet baseline | — | — | — | — | — | Not evaluated |
| Vision Transformer | — | — | — | — | — | Not evaluated |

Recommended comparison target: your prior `04-image-classification-resnet` project, provided both models are evaluated on the same data and environment.

## Evaluation artifacts

After evaluation, the scripts populate:

- `outputs/confusion_matrix.png`
- `outputs/classification_report.csv`
- `outputs/model_metrics.json`
- `outputs/macro_f1_results.json`
- `outputs/parameter_count_comparison.json`
- `outputs/vit_vs_cnn_latency_results.json`
- `outputs/vit_vs_cnn_comparison.csv`
- `outputs/sample_predictions.csv`
- `outputs/attention_visualization_examples.png`

## Local browser run

```bash
cd 08-image-classification-vision-transformer/web
python -m http.server 8000
```

Open `http://localhost:8000`. Do not open `index.html` with a `file://` URL because browser module and JSON loading restrictions can block the application.

## Python setup

```bash
cd 08-image-classification-vision-transformer
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-training.txt
pytest -q
```

Typical workflow:

```bash
python scripts/train_model.py --dataset cifar10 --epochs 3
python scripts/evaluate_model.py --checkpoint models/vit_model
python scripts/benchmark_latency.py --checkpoint models/vit_model
python scripts/compare_vit_cnn.py --vit-checkpoint models/vit_model --cnn-checkpoint models/cnn_or_resnet_baseline
python scripts/generate_attention_examples.py --checkpoint models/vit_model --image data/sample_images/example.jpg
python scripts/convert_to_onnx.py --checkpoint models/vit_model --output models/onnx_model
```

## GitHub Pages deployment

The repository workflow publishes the project under a subdirectory so other static demos can coexist:

```text
https://unit-mole.github.io/transformer-projects/08-image-classification-vision-transformer/
```

See [README_GITHUB_PAGES.md](README_GITHUB_PAGES.md) for exact configuration and troubleshooting.

## Project structure

```text
08-image-classification-vision-transformer/
├── README.md
├── README_GITHUB_PAGES.md
├── MODEL_CARD.md
├── DATASET_CARD.md
├── requirements.txt
├── requirements-training.txt
├── package.json
├── data/
├── notebooks/
├── src/
├── scripts/
├── tests/
├── models/
├── outputs/
├── images/
└── web/
```

## Portfolio positioning

**One-line description:** Browser-deployed Vision Transformer image classifier with client-side ONNX inference, top-k probabilities, latency benchmarking, explainability, and a reproducible ViT-vs-CNN evaluation framework.

**Pinned-repository description:** End-to-end Vision Transformer portfolio project: training/evaluation framework, ONNX browser deployment on GitHub Pages, patch-level explainability, model cards, CI, and ViT-vs-ResNet benchmarking.

**Skills demonstrated:** Vision Transformers, DeiT/ViT concepts, image preprocessing, model evaluation, macro F1, confusion matrices, ONNX export, Transformers.js, ONNX Runtime Web, static deployment, browser performance measurement, responsible AI documentation, testing, and GitHub Actions.

This project also connects naturally to quality analytics: the same architecture and evaluation discipline can be adapted to visual inspection, product/defect classification, image-based quality review, and production model-selection studies.

## Screenshots to add after deployment

1. Initial application screen with model-status panel.
2. Uploaded image and top prediction.
3. Top-five probability bars and measured latency.
4. Patch-sensitivity visualization with its disclaimer.
5. Real attention-rollout example generated from the final checkpoint.
6. ViT-vs-CNN evaluation table and confusion matrices after real evaluation.

## Limitations

- The live starter predicts ImageNet classes, not CIFAR-10 or Intel-scene labels.
- First model load depends on network speed and the Hugging Face Hub.
- WebGPU availability varies; the application falls back to WebAssembly.
- Prediction latency varies by browser, CPU/GPU, cache state, and image.
- Patch sensitivity is computationally expensive and is not equivalent to raw attention.
- No comparative metrics are published until real models are evaluated under matched conditions.
