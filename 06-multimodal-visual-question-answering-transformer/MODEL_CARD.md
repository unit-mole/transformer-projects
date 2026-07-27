# Model Card — Project 06 Multimodal VQA

## Models

### Local Python reference and evaluation model

- **Model:** `dandelin/vilt-b32-finetuned-vqa`
- **Architecture:** ViLT with a visual-question-answering classification head
- **Input:** RGB image and natural-language question
- **Output:** answer label logits
- **Confidence:** top softmax value and top-two margin, labeled as an uncalibrated proxy

### Static browser demo model

- **Model:** `Xenova/moondream2`
- **Architecture:** browser-compatible vision-language generative model
- **Runtime:** Transformers.js, ONNX Runtime Web, WebGPU
- **Preferred WebGPU profile:** fp16 embedding, fp16 vision encoder, q4 decoder
- **Compatibility WebGPU profile:** fp32 embedding, q8 vision encoder, q4 decoder
- **Confidence:** shown as `Not calibrated`; a reliable calibrated probability is not exposed by this implementation
- **Reliability behavior:** WebGPU preflight, automatic precision fallback, retryable worker reset, and populated failure states

## Intended use

Educational demonstrations, portfolio review, safe image understanding
experiments, and research prototyping.

## Not intended use

Medical, legal, financial, safety-critical, surveillance, biometric, identity,
security, employment, insurance, or official decision-making. Do not infer
sensitive personal attributes or identify real people.

## Training and fine-tuning

This repository does not claim to have trained or fine-tuned either base model.
It provides pretrained inference, preprocessing, evaluation, and deployment
infrastructure.

## Evaluation

Supported metrics include VQA consensus accuracy, exact match, question-type
accuracy, category analysis, manual review, failure analysis, and latency.
Committed metric files remain `not_evaluated` until the scripts are executed.

## Limitations and risks

Possible failures include incorrect objects, colors, counts, actions, spatial
relations, OCR, small objects, poor-quality images, ambiguous questions, and
out-of-distribution scenes. Browser performance depends on hardware, memory,
drivers, WebGPU support, and network speed.

## Privacy

Use only non-sensitive images. The static app performs inference in the browser,
but it downloads model components from the Hugging Face Hub and may cache them
locally. Do not upload private or confidential material.
