---
title: Long Document QA Browser
emoji: 📄
colorFrom: blue
colorTo: green
sdk: static
app_build_command: npm install --no-audit --no-fund && npm run build
app_file: dist/index.html
pinned: false
license: mit
models:
  - Xenova/distilbert-base-cased-distilled-squad
---

# Long-Document QA — Browser Deployment Baseline

This is the free, static deployment layer for Transformer Portfolio Project 04.
It runs real extractive question-answering inference in the visitor's browser
using Transformers.js, ONNX Runtime, and the browser-compatible model:

```text
Xenova/distilbert-base-cased-distilled-squad
```

## Important architecture disclosure

The complete GitHub project uses:

```text
valhalla/longformer-base-4096-finetuned-squadv1
```

The Static Space does **not** claim to run Longformer. Longformer and BigBird are
not listed among the currently supported Transformers.js architectures, and a
full Longformer browser deployment would be unnecessarily large and fragile for
this portfolio demo. Instead, the static application demonstrates a transparent
browser baseline:

```text
Long document
  → overlapping chunks
  → lexical candidate retrieval
  → DistilBERT extractive QA over the best chunks
  → answer aggregation
  → supporting paragraph and highlighted evidence
```

This separation preserves technical honesty while still providing a permanently
free, interactive portfolio demonstration.

## Local development

```bash
npm install
npm run dev
```

## Validation

```bash
npm test
npm run check
npm run build
```

## Deploy as a Hugging Face Static Space

1. Create a new **Static HTML** Space.
2. Copy the contents of this `web/` directory to the Space repository root.
3. Commit and push.
4. Hugging Face reads the YAML above, runs the Vite build, and serves
   `dist/index.html`.

Replace the placeholder GitHub, model-card, Gradio, and Static Space links in
`src/config.js` before deployment.

## Responsible use

Do not upload confidential, proprietary, sensitive, copyrighted, regulated, or
personally identifiable documents. The model can produce incorrect or
unsupported answers. The confidence score is an uncalibrated proxy, not a
correctness guarantee. Human review is required.

## Official references

- https://huggingface.co/docs/hub/spaces-sdks-static
- https://huggingface.co/docs/transformers.js/index
- https://huggingface.co/docs/transformers.js/pipelines
- https://huggingface.co/Xenova/distilbert-base-cased-distilled-squad
