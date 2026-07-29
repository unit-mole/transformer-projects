# Changelog

## 1.0.0 — Portfolio rebuild

- Replaced the 263-cell synthetic-first notebook as the production source of truth with modular Python packages.
- Preserved useful ideas: public dataset loading, safe samples, baselines, output bundling, and latency measurement.
- Removed 75 repeated analysis/checkpoint blocks.
- Made the Transformer model the real inference path instead of an optional disabled branch.
- Added direct model loading compatible with the modern Transformers API.
- Added Gradio, full generation controls, long-text chunking, ROUGE, BERTScore, honest output templates, LSTM comparison, tests, CI, Docker, model card, and deployment guides.

## 1.1.0 — Free Static Space deployment

- Added a complete `web/` application using Transformers.js and ONNX Runtime Web.
- Added WebGPU and WASM runtime selection with automatic fallback.
- Added model-loading progress, token-aware chunking, beam comparison, latency, compression, token, and chunk metrics.
- Added Static Space metadata and Vite build configuration.
- Extended GitHub Actions to test/build the browser app and optionally synchronize it to Hugging Face.
- Updated deployment documentation and model metadata without removing the Python/Gradio implementation.

## RTX fine-tuning and benchmark upgrade

Added the complete GPU notebook, deterministic benchmark, baselines, pretrained/fine-tuned comparison, optional LSTM import, actual metrics, error analysis, charts, artifact promotion, validation, and optional Model Hub publication.
