# Model Card — CLIP Image-Text Retrieval and Zero-Shot Classification

## Model

- **Base model:** `openai/clip-vit-base-patch32`
- **Browser-compatible model repository:** `Xenova/clip-vit-base-patch32`
- **Architecture:** CLIP text encoder + Vision Transformer image encoder + projection heads
- **Embedding dimension:** 512
- **Similarity metric:** cosine similarity on L2-normalized embeddings
- **Browser runtime:** Transformers.js backed by ONNX Runtime Web
- **Default dtype:** quantized `q8` when available
- **Default execution provider:** WebAssembly for broad compatibility

## Tasks

1. Text-to-image retrieval over a fixed gallery.
2. Zero-shot image classification from user-defined candidate labels.

## Inputs

- Text query: natural-language scene or object description.
- Image: RGB PNG/JPEG/WebP loaded from the gallery or uploaded by the user.
- Candidate labels: user-editable terms converted to prompts using `a photo of a {label}`.

## Outputs

- Normalized text and image embeddings.
- Ranked images with cosine similarity.
- Ranked zero-shot labels with cosine similarity and candidate-set softmax values.
- Browser latency measurements.

## Intended use

Education, portfolio demonstrations, safe image-gallery exploration, prototyping visual knowledge retrieval, and human-reviewed research experiments.

## Not intended for

Medical, legal, financial, safety-critical, surveillance, biometric, identity, hiring, insurance, security, quality-release, or official decision-making. The model must not be used to infer sensitive personal attributes or identify real people.

## Evaluation

The repository supports Recall@1, Recall@5, Recall@10, score-distribution analysis, prompt-sensitivity analysis, manual relevance review, and latency benchmarking. Metrics are intentionally not pre-filled. Run the included evaluation scripts and report only measured values.

## Quantization and deployment

The GitHub Pages app uses browser-compatible quantized ONNX assets loaded through Transformers.js. Gallery image embeddings can be generated offline and committed as JSON, or generated once in the browser and cached. Large model binaries are not included in this archive.

## Limitations and risks

- CLIP can produce biased or misleading associations.
- Results depend on gallery coverage, prompt phrasing, and image quality.
- Similarity is not a calibrated probability.
- Zero-shot softmax is relative to the candidate-label set.
- Quantization can slightly change scores.
- Synthetic gallery assets are not a production benchmark.
- Browser performance depends on memory, CPU/GPU support, and network conditions.

## Privacy

Uploaded images are processed locally in the browser by the application code. Users must still avoid uploading private, confidential, proprietary, copyrighted, medical, identity, or personally identifiable images to a public demo.
