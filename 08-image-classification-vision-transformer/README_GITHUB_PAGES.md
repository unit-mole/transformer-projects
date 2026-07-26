# GitHub Pages Deployment Guide

This project is designed as a static browser application. GitHub Pages serves only HTML, CSS, JavaScript, JSON, SVG, and optional local model assets; inference runs in the visitor's browser.

## Recommended repository URL

```text
https://github.com/unit-mole/transformer-projects
```

## Published project URL

```text
https://unit-mole.github.io/transformer-projects/08-image-classification-vision-transformer/
```

## Deployment architecture

The workflow `.github/workflows/08-image-classification-vision-transformer.yml`:

1. Runs tests and validates required files.
2. Creates `_site/`.
3. Copies `pages/` to the root of `_site/`.
4. Copies `08-image-classification-vision-transformer/web/` to `_site/08-image-classification-vision-transformer/`.
5. Uploads `_site/` as the GitHub Pages artifact.
6. Deploys on pushes to `main`.

This allows future browser projects to be added as sibling subdirectories.

## One-time GitHub configuration

1. Push the repository to GitHub.
2. Open **Settings → Pages**.
3. Under **Build and deployment**, select **GitHub Actions** as the source.
4. Open the **Actions** tab and run or re-run `08 Image Classification Vision Transformer CI and Pages`.
5. Wait for the deploy job to complete.
6. Open the deployment URL shown in the workflow summary.

## Test locally before publishing

```bash
cd 08-image-classification-vision-transformer/web
python -m http.server 8000
```

Open `http://localhost:8000` and verify:

- the model status changes from `Not loaded` to `Ready`;
- the first model download completes;
- uploaded and sample images render;
- top-k predictions appear;
- latency is shown;
- the patch-sensitivity map runs;
- the browser console has no 404/CORS errors.

## Default directly deployable model

`web/metadata.json` points to:

```text
onnx-community/vit-tiny-patch16-224-ONNX
```

Transformers.js downloads a quantized ONNX file and its configuration from the Hugging Face Hub. This avoids committing a large binary to ordinary Git history and makes the static project work immediately after Pages deployment.

## Use your own fine-tuned model

1. Train/fine-tune the model in Python.
2. Export it with `scripts/convert_to_onnx.py` or Optimum.
3. Convert the repository to a Transformers.js-compatible local folder, including `config.json`, `preprocessor_config.json`, and the `onnx/` directory.
4. Copy that folder under `web/model/`.
5. Update `web/metadata.json`:

```json
{
  "browser_model": {
    "source": "local",
    "model_id": "./model",
    "dtype": "q8"
  }
}
```

6. In `web/inference.js`, set `env.allowLocalModels = true` and `env.localModelPath = './'` if your exported folder requires local-model resolution.
7. Test with `python -m http.server` before pushing.

Do not commit oversized model files without checking GitHub's file limits. Prefer quantization and Git LFS when appropriate.

## Common problems

### `Failed to fetch metadata.json`

Run through an HTTP server instead of opening the file directly. Confirm that `metadata.json` is beside `index.html`.

### Model download stalls

Check the browser network panel, disable restrictive extensions for the test, and retry. The first load is slower; later loads may use browser cache.

### WebGPU error

The code automatically retries with WebAssembly. WebGPU is an acceleration option, not a deployment requirement.

### 404 under the Pages subdirectory

Use relative paths beginning with `./`, not root-absolute paths beginning with `/`. This project already follows that rule.

### Changes do not appear

Confirm that the Pages source is GitHub Actions, verify the deploy job succeeded, then hard-refresh or clear the site cache.

### Model labels do not match the final dataset

Replace the default ImageNet starter model with the actual fine-tuned checkpoint and update model/dataset cards. Do not simply replace `class_names.json` while leaving a different classification head in the model.
