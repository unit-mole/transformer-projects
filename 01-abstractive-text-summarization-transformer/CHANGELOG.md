# Changelog

## 1.0.0 — Portfolio rebuild

- Replaced the 263-cell synthetic-first notebook as the production source of truth with modular Python packages.
- Preserved useful ideas: public dataset loading, safe samples, baselines, output bundling, and latency measurement.
- Removed 75 repeated analysis/checkpoint blocks.
- Made the Transformer model the real inference path instead of an optional disabled branch.
- Added direct model loading compatible with the modern Transformers API.
- Added Gradio, full generation controls, long-text chunking, ROUGE, BERTScore, honest output templates, LSTM comparison, tests, CI, Docker, model card, and deployment guides.
