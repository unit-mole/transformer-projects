# Project 01 — Free Hugging Face Static Space Deployment

## Why a separate browser layer is required

A Static Space serves HTML, CSS, and JavaScript. It does not execute `app.py`, Gradio, PyTorch, or the Python `transformers` package. Therefore, the Python application remains in the GitHub repository while `web/` provides equivalent browser inference using Transformers.js and ONNX Runtime Web.

## Space identity

Recommended Space:

```text
Owner: anmol-unitmole
Space name: 01-abstractive-text-summarization-transformer
Full ID: anmol-unitmole/01-abstractive-text-summarization-transformer
SDK: Static
License: MIT
Visibility: Public
```

## Automatic deployment from GitHub

### 1. Create a Hugging Face token

Create a user access token with write permission. Never commit it to the repository.

### 2. Add GitHub Actions configuration

Open the GitHub repository and go to:

```text
Settings → Secrets and variables → Actions
```

Add a **secret**:

```text
Name: HF_TOKEN
Value: <your Hugging Face write token>
```

Add a **variable**:

```text
Name: HF_SPACE_REPO
Value: anmol-unitmole/01-abstractive-text-summarization-transformer
```

### 3. Push Project 01

The Project 01 workflow will:

1. run Python tests and import checks;
2. install and test the JavaScript frontend;
3. build the Static Space with Vite;
4. create the Space as `static` when it does not exist;
5. replace the Space files with the contents of `web/`.

If `HF_SPACE_REPO` is not configured, the synchronization job is intentionally skipped while all CI tests still run.

## Manual deployment

Create a new Space in the Hugging Face UI and select:

```text
SDK: Static
Template: Blank or Transformers.js
```

Then upload the **contents** of `web/`, not the `web` folder itself, so the Space root contains:

```text
README.md
index.html
package.json
vite.config.js
public/
src/
tests/
```

The YAML block in `web/README.md` instructs Hugging Face to execute:

```text
npm run build
```

and serve:

```text
dist/index.html
```

## Local validation

```bash
cd web
npm install
npm test
npm run build
npm run preview
```

## Model loading behavior

The first visitor downloads quantized ONNX files from `Xenova/distilbart-cnn-12-6`.

- WebGPU path: `q4f16`
- WASM path: `q8`
- Cache: browser cache managed by Transformers.js

The model is large enough that first load can be slow. This is a model-size limitation, not paid-server usage.

## Common issues

### Synchronization job is skipped

Confirm that the repository variable `HF_SPACE_REPO` exists and the workflow is running on `main`.

### Synchronization job fails with authentication error

Confirm that `HF_TOKEN` is a repository secret with write permission and that the token owner can create or update the target Space.

### Static build fails

Run `npm install`, `npm test`, and `npm run build` from `web/` locally. Confirm the pinned package versions in `package.json` have not been changed unintentionally.

### WebGPU fails

Choose WASM in the app. Auto mode already tries WebGPU and then falls back to WASM.

### First load appears slow

Keep the page open and monitor the progress panel. The application must download hundreds of megabytes of ONNX weights on first use. Later loads should benefit from browser caching.
