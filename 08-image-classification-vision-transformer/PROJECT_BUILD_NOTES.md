# Build notes

## Decisions made

- Repository name follows the supplied GitHub screenshot: `transformer-projects`.
- The app is static and deploys under a project-specific Pages subdirectory.
- The live starter uses a compact quantized ONNX model from the Hugging Face Hub, avoiding a Python backend and oversized Git binary.
- Transformers.js handles the exact processor configuration associated with the model.
- Live explainability is correctly labeled patch sensitivity; raw attention is not claimed.
- Python code includes a genuine attention-rollout path for the final compatible checkpoint.
- Evaluation files contain null values until real models are supplied and evaluated.

## Files intentionally not bundled

- A custom trained CIFAR-10/Intel checkpoint.
- A CNN/ResNet baseline checkpoint.
- Full datasets.
- Fabricated evaluation charts or metrics.
- A fake attention heatmap.

## Finalization checklist

1. Fine-tune or supply the actual checkpoint.
2. Evaluate it on the documented test split.
3. Add the matched CNN/ResNet metrics.
4. Export and validate the browser model.
5. Generate a real attention-rollout example.
6. Replace the screenshot placeholder after Pages deployment.
