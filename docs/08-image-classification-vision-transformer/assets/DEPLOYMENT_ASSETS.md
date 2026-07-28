# Project 08 Deployment Assets

Generated: 2026-07-27T20:46:42.223752+00:00

## Browser Model

- File: `models/model_browser.onnx`
- Precision: `float32`
- Size: `23.32 MB`
- Input: `pixel_values`
- Output: `logits`
- Classes: `10`
- Execution-provider priority: WebGPU, then WASM

## Application Data

The consolidated frontend data file is:

`data/app_data.json`

## Validation

- ONNX graph checker: Passed
- Dynamic batch support: Passed
- Deployment parity review: Accepted
- Test accuracy: 96.45%
- Macro F1: 0.9644
- PyTorch/ONNX prediction agreement: 99.99%

## Important Limitation

This model predicts only the following CIFAR-10 classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Images outside the CIFAR-10 distribution may produce unreliable results.
