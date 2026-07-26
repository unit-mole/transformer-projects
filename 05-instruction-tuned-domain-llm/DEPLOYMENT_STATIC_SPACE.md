# Hugging Face Static Space Deployment

This guide deploys the `web/` application as a Hugging Face **Static Space**. The model runs in the visitor's browser through Transformers.js and ONNX Runtime Web.

## Prerequisites

- A public Hugging Face account
- Node.js 20 or newer for local validation
- A browser-compatible ONNX model repository, or the included base-model fallback

## Test locally

```bash
cd web
npm install
npm run validate
npm run dev
```

Open the local Vite address shown in the terminal. On first inference, the browser downloads and caches model files.

## Build locally

```bash
npm run build
npm run preview
```

## Create the Space

1. Create a new Hugging Face Space.
2. Select **Static** as the SDK.
3. Use a public repository for a recruiter-facing portfolio demo.
4. Upload the contents of the `web/` folder to the root of the Space repository.
5. Keep `README.md`, `index.html`, `package.json`, `vite.config.js`, `public/`, `scripts/`, and `src/` at the Space root.
6. Push the files. Hugging Face will run `npm run build` and serve `dist/index.html` according to the README metadata.

## Base-model mode

The app works immediately with:

```text
Xenova/flan-t5-small
```

This is a browser-compatible ONNX conversion of `google/flan-t5-small`. The interface clearly identifies it as the base model, not as a model trained by this project.

## Domain-model mode

After training and exporting your own model:

1. Merge the LoRA adapter into FLAN-T5.
2. Export the merged checkpoint to ONNX.
3. Quantize the ONNX files.
4. Upload the browser-model directory to a Hugging Face model repository.
5. In the live app, select **Custom merged domain model** and enter the repository ID.

The model repository must contain tokenizer and configuration files at its root and ONNX weights in an `onnx/` subfolder.

## Recommended repository names

```text
YOUR_HF_USERNAME/ml-ds-instruction-tuned-flan-t5-small-lora
YOUR_HF_USERNAME/ml-ds-instruction-tuned-flan-t5-small-onnx
YOUR_HF_USERNAME/ml-ds-instruction-tuned-assistant
```

## Important limitations

- The first load can be slow because model files are downloaded into the browser cache.
- WebGPU availability varies by browser and device; the app falls back to WASM.
- Small FLAN-T5 models can generate incomplete or inaccurate explanations.
- Static deployment cannot dynamically attach a PyTorch PEFT adapter. The adapter must be merged before ONNX export.
- Never claim the browser model is fine-tuned until you have trained, merged, exported, tested, and published it.
