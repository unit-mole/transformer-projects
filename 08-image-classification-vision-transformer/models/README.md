# Model artifacts

Large checkpoints and ONNX binaries are ignored by default.

Expected locations:

- `vit_model/` — final Python checkpoint and processor configuration.
- `cnn_or_resnet_baseline/` — matched baseline checkpoint.
- `onnx_model/` — exported ONNX/Transformers.js-compatible model.
- `tfjs_model/` — optional TensorFlow.js export.

The directly deployable web starter downloads a small quantized ONNX model from the Hugging Face Hub instead of committing it here.
