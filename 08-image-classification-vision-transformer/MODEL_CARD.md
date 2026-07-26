# Model Card — Compact Vision Transformer Browser Classifier

## Model overview

| Field | Value |
|---|---|
| Portfolio model name | Project 08 Compact ViT Browser Classifier |
| Default deployed model | `onnx-community/vit-tiny-patch16-224-ONNX` |
| Original model family | Vision Transformer, tiny, patch size 16, 224×224 input |
| Task | Single-label image classification |
| Default label space | ImageNet-1k |
| Runtime | Transformers.js on ONNX Runtime Web |
| Quantization | `q8` requested by default; runtime availability determines exact asset |
| Deployment | GitHub Pages, client-side inference |

## Intended use

Educational demonstrations, portfolio review, browser-AI experimentation, model deployment learning, and non-critical image-classification prototypes using safe public or synthetic images.

## Not intended for

Medical, legal, financial, surveillance, identity recognition, sensitive-attribute inference, security, hiring, insurance, product-release, quality-release, or other high-impact decisions. It must not be used as a substitute for human review.

## Inputs and preprocessing

The browser model's own preprocessor configuration is loaded by Transformers.js. The starter uses 224×224 RGB input, rescaling, and the normalization values defined by the model repository. The project metadata records `[0.5, 0.5, 0.5]` mean and standard deviation for documentation, but the runtime model configuration remains the source of truth.

## Training and fine-tuning

The default deployed model is pretrained inference. This repository does not claim that the author fine-tuned it. The Python framework supports fine-tuning a compact ViT on CIFAR-10 or another documented dataset; update this card when an actual checkpoint is produced.

## Conversion

The browser starter is an ONNX community conversion. For a custom model, use Optimum/ONNX export and validate predictions against the Python checkpoint before deployment.

## Evaluation status

| Metric | Value |
|---|---:|
| Accuracy | Not evaluated in this repository |
| Macro F1 | Not evaluated in this repository |
| Confusion matrix | Not generated |
| Parameter count | Populate from the final checkpoint |
| Python latency | Not benchmarked |
| Browser latency | Measured per user prediction, device-dependent |
| Comparative CNN latency | Not benchmarked |

Metrics must be computed on a named dataset split. Do not copy pretrained-model benchmark numbers into a project evaluation table unless the same checkpoint, preprocessing, and evaluation protocol are verified.

## Explainability

The live interface offers occlusion-based patch sensitivity. It does not call this raw attention. The Python attention module can generate actual class-token attention rollout if the final model exposes attention tensors.

## Risks and limitations

The model can be wrong on blurry, cropped, adversarial, out-of-distribution, synthetic, or domain-specific images. Confidence may be poorly calibrated. ImageNet labels can be overly specific, outdated, or unsuitable for the user's domain. Browser performance varies, and the remote model must be downloaded on first use.

## Privacy

The application does not intentionally upload user images; inference runs locally in the browser. Nevertheless, users should avoid private, sensitive, proprietary, copyrighted, or personally identifying images. Browser extensions, device policies, or third-party infrastructure are outside the project's control.

## License

Repository code: MIT. Model and dataset assets retain their original licenses and terms.
