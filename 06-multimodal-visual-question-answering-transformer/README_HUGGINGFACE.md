# Hugging Face Static Space deployment guide

## Space creation values

- **Owner:** `anmol-unitmole`
- **Space name:** `06-multimodal-visual-question-answering-transformer`
- **Short description:** Browser-based multimodal VQA with SmolVLM, WebGPU, generation-confidence diagnostics, and a 60-pair evaluation lab.
- **License:** MIT
- **SDK:** Static
- **Template:** Blank
- **Visibility:** Public

Upload the **contents** of `space/` to the root of the Hugging Face Space. Do
not upload the outer Project 06 folder or an extra nested `space` directory.

## Required Space structure

```text
README.md
index.html
samples/
evaluation/
│   ├── images/
│   ├── vqa_evaluation_60.csv
│   └── vqa_evaluation_60.json
src/
│   ├── main.js
│   ├── model-worker.js
│   └── style.css
```

The root README metadata must include:

```yaml
---
title: Multimodal Visual Question Answering Transformer
emoji: 🖼️
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
models:
  - HuggingFaceTB/SmolVLM-256M-Instruct
---
```

## What the deployed site provides

- image upload and safe sample images;
- SmolVLM-256M-Instruct WebGPU inference;
- predicted answer, question type, answer type, and latency;
- generation confidence proxy when token scores are available;
- explicit statement that token scores are not calibrated correctness probabilities;
- 60-pair browser evaluation;
- overall and category-wise accuracy;
- answer failure rate and latency statistics;
- downloadable JSON results and manual failure review.

## GitHub Actions synchronization

Create these settings in the GitHub repository:

- secret `HF_TOKEN`: a Hugging Face user access token with write permission;
- variable `HF_SPACE_REPO`: `anmol-unitmole/06-multimodal-visual-question-answering-transformer`.

A push affecting Project 06 validates the Python utilities, the balanced
60-pair evaluation suite, JavaScript syntax, required Static Space files, and
large-file limits. The final job synchronizes `space/` to the Hugging Face
Space.

## Final verification

After the workflow succeeds:

1. Open the Space page.
2. Confirm the model name is `HuggingFaceTB/SmolVLM-256M-Instruct`.
3. Run the Shapes example with `What color is the circle?`.
4. Confirm an answer, answer-confidence diagnostic, and latency are populated.
5. Run the 60-question evaluation once on the device you plan to document.
6. Download the JSON report and save it under `outputs/` only after reviewing it.
7. Add the Space page URL to GitHub and your portfolio.
