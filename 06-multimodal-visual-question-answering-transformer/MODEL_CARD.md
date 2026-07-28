# Model Card — Project 06 Multimodal VQA

## Static browser model

- **Model:** `HuggingFaceTB/SmolVLM-256M-Instruct`
- **Task:** image-text-to-text / visual question answering
- **Runtime:** Transformers.js, ONNX Runtime Web, WebGPU
- **Precision:** stable WebGPU `fp32`
- **Inputs:** RGB image and natural-language question
- **Outputs:** generated text answer, optional generation-score diagnostic, and latency

## Local Python reference model

- **Model:** `dandelin/vilt-b32-finetuned-vqa`
- **Architecture:** ViLT with a VQA classification head
- **Purpose:** local reproducible inference and classification-style evaluation experiments

## Answer-confidence interpretation

The browser implementation requests `output_scores` together with
`return_dict_in_generate`. When per-step score tensors are available, it
computes the geometric mean of the selected-token probabilities and labels the
result a **generation confidence proxy**.

This value is not calibrated. It measures how strongly the decoder preferred
its generated tokens, not the probability that the visual answer is correct.
When scores are unavailable, the interface displays:

```text
Not available for this generative model
```

A calibrated correctness estimate would require a held-out labeled dataset,
correct/incorrect outcome labels, a calibration model, and calibration metrics
such as expected calibration error and Brier score.

## Evaluation

The browser includes a 60-pair synthetic portfolio suite with 10 questions in
each category:

- color;
- object identification;
- counting;
- yes/no;
- action or scene;
- spatial relationship.

Reported outputs include overall accuracy, category-wise accuracy, answer
failure rate, average latency, latency range, individual predictions, and a
failure-analysis preview. These results are not an official VQA v2 benchmark.

## Intended use

Educational demonstrations, portfolio review, safe multimodal experiments,
browser inference research, and prototyping.

## Not intended use

Medical, legal, financial, safety-critical, surveillance, biometric, identity,
security, employment, insurance, or official decision-making. Do not identify
real people or infer sensitive personal attributes.

## Training and fine-tuning

This repository does not claim to have trained or fine-tuned either model. It
provides pretrained inference, preprocessing, evaluation, and deployment
infrastructure.

## Limitations and risks

The model can return incorrect objects, colors, counts, actions, and spatial
relations. It may hallucinate, struggle with OCR or small objects, and fail on
ambiguous, low-quality, or out-of-distribution images. Browser performance
depends on hardware, memory, graphics drivers, browser version, WebGPU support,
and network speed.

## Privacy

Use only non-sensitive images. The Static Space performs inference in the
browser, but downloads and caches model components from the Hugging Face Hub.
Do not upload private, confidential, medical, identity, workplace, or otherwise
sensitive images.
