# Deploy Project 02 as a Free Hugging Face Static Space

## What is deployed

Upload the **contents** of `web/` to the root of a separate Hugging Face Space.
The Space runs real ONNX MarianMT inference in the browser with Transformers.js.
The Python files remain in GitHub and are not executed by the Static Space.

## Create the Space

1. Open Hugging Face and create a new Space.
2. Choose **Static** as the SDK.
3. Select **Blank** or **Transformers.js** as the template.
4. Use a name such as `english-hindi-neural-machine-translation`.
5. Choose the MIT license and create the Space.

## Upload from the web folder

From inside `02-neural-machine-translation-transformer/web`:

```bash
git init
git branch -M main
git remote add origin https://huggingface.co/spaces/<HF_USERNAME>/<SPACE_NAME>
git add .
git commit -m "Deploy Project 02 Static Transformers.js demo"
git push -u origin main
```

Use a Hugging Face write token when Git asks for a password.

## Important behavior

- The first request downloads a quantized directional model from the Hub.
- Only the requested direction is loaded.
- Browser cache makes later requests faster.
- WASM/browser inference can be slower than server inference.
- The app limits CSV runs to 25 rows to protect browser memory.
- The confidence value is an explainable heuristic, not a calibrated probability.

## Models

- `Xenova/opus-mt-en-hi`
- `Xenova/opus-mt-hi-en`

These repositories contain ONNX-compatible versions of the corresponding
Helsinki-NLP MarianMT models.
