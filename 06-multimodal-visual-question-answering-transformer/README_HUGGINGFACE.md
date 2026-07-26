# Hugging Face Static Space deployment guide

## Space creation values

- **Owner:** `anmol-unitmole`
- **Space name:** `06-multimodal-visual-question-answering-transformer`
- **Short description:** Browser-based visual question answering with a vision-language Transformer, Transformers.js, ONNX, and WebGPU.
- **License:** MIT
- **SDK:** Static
- **Template:** Blank
- **Visibility:** Public

Upload the **contents** of `space/` to the root of the Hugging Face Space. Do
not upload the outer `space` directory as a nested folder.

## Required files in the Space root

```text
README.md
index.html
src/main.js
src/model-worker.js
src/style.css
samples/shapes_scene.png
samples/three_blocks.png
samples/yellow_triangle.png
```

The Space is compute-free because inference runs in the visitor's browser.
WebGPU and a large first-time model download are required. No Hugging Face token
should be embedded in JavaScript.

## GitHub Actions synchronization

Add repository secret `HF_TOKEN` and repository variable `HF_SPACE_REPO`.
Pushing changes under Project 06 will validate Python utilities and static files,
then synchronize the `space/` folder to the Space.
