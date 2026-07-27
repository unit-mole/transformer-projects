# Browser model assets

The default app loads `Xenova/clip-vit-base-patch32` with quantized `q8` ONNX weights through Transformers.js. Large ONNX binaries are intentionally not committed.

For a fully self-hosted deployment, copy a complete Transformers.js-compatible model repository layout into this folder, then set `env.allowRemoteModels = false`, `env.allowLocalModels = true`, `env.localModelPath = './model/'`, and update `MODEL_ID` in `clip_inference.js` to the local directory name.

Do not add fake `.onnx` placeholders. Only commit valid generated or downloaded model binaries.
