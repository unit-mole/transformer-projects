# Project 05 — Static Browser Deployment Roadmap

**Static deployment does not reduce the value of Project 05 as an instruction-tuned Transformer portfolio project.** It changes where inference runs:

- **Python / Gradio:** FLAN-T5 and an optional LoRA adapter run through PyTorch on hosted compute.
- **Static + Transformers.js:** a merged, ONNX-converted FLAN-T5 model runs inside the visitor's browser through ONNX Runtime Web.

Both paths perform real Transformer inference. The static application is not a mock chatbot and does not call a hidden Python server.

## Best portfolio approach

Use a three-part structure:

| Portfolio component | Purpose |
|---|---|
| GitHub repository | Full Python training, LoRA / PEFT, evaluation, notebooks, tests, CI, Gradio, and export tooling |
| Hugging Face model repositories | Publish the LoRA adapter and the merged ONNX browser model with model cards |
| Hugging Face Static Space | Host the live browser-based ML/Data Science Learning Assistant |

This structure demonstrates both model adaptation and deployment engineering.

## 1. Keep the complete Python project

Do not remove the existing Python implementation:

```text
app.py
gradio_app.py
configs/
data/
notebooks/
src/
scripts/
tests/
outputs/
requirements.txt
MODEL_CARD.md
DATASET_CARD.md
```

These files demonstrate:

- FLAN-T5 encoder-decoder architecture
- custom instruction-dataset preparation
- LoRA / PEFT configuration and adapter training
- base-model versus adapter comparison
- instruction-adherence evaluation
- BERTScore and response-relevance analysis
- hallucination review and manual evaluation
- Gradio application engineering
- automated tests and GitHub Actions

The Python implementation remains the technical foundation of the project.

## 2. Add a dedicated Static frontend

The project now includes:

```text
05-instruction-tuned-domain-llm/
│
├── app.py
├── src/
├── scripts/
├── tests/
├── notebooks/
├── outputs/
│
└── web/
    ├── README.md
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── public/
    │   ├── model-metadata.json
    │   ├── evaluation-summary.json
    │   └── architecture.svg
    ├── scripts/
    │   └── validate-config.mjs
    └── src/
        ├── main.js
        ├── model-client.js
        ├── model-worker.js
        ├── prompt-templates.js
        ├── examples.js
        ├── evaluation.js
        ├── config.js
        ├── utils.js
        └── styles.css
```

The frontend uses Transformers.js with a text-to-text generation pipeline. It loads ONNX model weights from the Hugging Face Hub and performs inference locally in the browser.

## 3. What the Static demo displays

The browser application includes:

- real browser-based Transformer inference notice
- base model and custom merged-domain-model selection
- ONNX model repository input
- model-loading progress
- WebGPU detection with WASM fallback
- quantization selection
- prompt-category selector
- curated sample prompts
- optional supporting context
- maximum-token, temperature, top-p, and repetition controls
- actual tokenizer output preview
- source and generated token counts
- inference latency
- model, device, and dtype metadata
- clearly labelled demo-only adherence and relevance heuristics
- hallucination-risk indicators
- architecture explanation
- evaluation-status section
- limitations and responsible-use guidance
- GitHub, model-card, dataset-card, and evaluation links

## 4. Required model publication flow

LoRA adapters cannot simply be attached to a browser pipeline at runtime. For the Static Space, use this publication flow:

```text
Base FLAN-T5-small
        +
Trained LoRA adapter
        ↓
Merge adapter into the base model
        ↓
Export the merged model to ONNX
        ↓
Create quantized browser weights
        ↓
Upload the web-ready model repository to Hugging Face
        ↓
Load it from Transformers.js in the Static Space
```

The repository includes scripts for the first four steps:

```bash
python scripts/merge_lora_adapter.py \
  --adapter models/lora_adapter \
  --output models/merged_model

python scripts/export_merged_model_to_onnx.py \
  --model models/merged_model \
  --output models/browser_model \
  --quantize
```

After verifying the ONNX model, publish `models/browser_model/` to a Hugging Face model repository and enter that repository ID in the Static demo.

## 5. Recommended Hugging Face repositories

Create these only after the artifacts genuinely exist:

```text
YOUR_HF_USERNAME/ml-ds-instruction-tuned-flan-t5-small-lora
YOUR_HF_USERNAME/ml-ds-instruction-tuned-flan-t5-small-onnx
```

Do not label the original `google/flan-t5-small` or `Xenova/flan-t5-small` checkpoint as your fine-tuned model. Until your adapter is trained and exported, the static application transparently uses the browser-compatible base model for demonstration.

## 6. Final presentation

```text
GitHub
└── Complete Python + JavaScript ML engineering project

Hugging Face Model Hub
├── LoRA adapter repository
└── Merged ONNX browser-model repository

Hugging Face Static Space
└── Live Transformers.js ML/Data Science Learning Assistant
```

This final setup demonstrates:

- Transformer encoder-decoder inference
- instruction tuning
- LoRA and PEFT
- adapter merging
- ONNX export and quantization
- browser-side model inference
- WebGPU / WASM deployment
- custom curriculum design
- evaluation and hallucination analysis
- responsible AI documentation
- Python, JavaScript, CI, and deployment engineering
