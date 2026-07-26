# Browser Deployment Upgrade

This upgrade preserves the existing Python Longformer project and adds a separate
free browser deployment baseline.

## Added

- `web/` Vite application for Hugging Face Static Spaces;
- Transformers.js and ONNX extractive-QA inference;
- TXT, Markdown, CSV, and selectable-text PDF parsing in the browser;
- overlapping long-document chunking;
- lexical candidate-chunk retrieval;
- best-answer aggregation across candidate chunks;
- supporting-paragraph mapping and escaped evidence highlighting;
- WASM and optional WebGPU runtime selection;
- model-download and inference progress displays;
- browser diagnostics and downloadable JSON output;
- JavaScript unit tests and syntax checks;
- browser-baseline model card;
- Static Space deployment instructions;
- a dual-deployment roadmap;
- GitHub Actions browser validation alongside the Python test job.

## Preserved

- Longformer PyTorch inference pipeline;
- Gradio application;
- evaluation scripts and notebooks;
- Python tests;
- model card and responsible-use notes;
- sample data and output structure.

## Technical honesty

The browser baseline uses
`Xenova/distilbert-base-cased-distilled-squad`. It does not claim to run
Longformer. The full Python implementation continues to use
`valhalla/longformer-base-4096-finetuned-squadv1`.
