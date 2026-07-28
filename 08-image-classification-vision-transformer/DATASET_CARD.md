# Dataset Card — CIFAR-10 Training Framework and ImageNet Starter

## Dataset roles

This project separates two roles:

1. **Directly deployable starter:** a pretrained ImageNet-1k compact ViT used for immediate GitHub Pages inference.
2. **Portfolio training framework:** CIFAR-10, loaded programmatically by the training scripts and never committed in full.

## CIFAR-10 summary

| Field | Value |
|---|---|
| Purpose | Reproducible ten-class image-classification training and ViT-vs-CNN comparison |
| Format | 32×32 RGB images with integer class labels |
| Classes | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Public repository content | Class metadata and a small safe sample folder only |
| Full data | Downloaded by `torchvision` into ignored local storage |
| Model input | Resized to the selected model's input resolution, typically 224×224 |

## Required split documentation

When you run the final experiment, record:

- exact train/validation/test counts;
- random seed;
- stratification method;
- augmentation configuration;
- image resizing and normalization;
- duplicate/leakage checks;
- class distribution;
- dataset version and source;
- any removed or corrupted records.

## ImageNet starter

The live starter model uses the label mapping and preprocessing configuration included with its pretrained model repository. It is not evaluated as a CIFAR-10 model, and the README does not present it as one.

## Sensitive data and public samples

Use only safe, non-sensitive, appropriately licensed samples. Do not publish personal photos, IDs, medical images, confidential workplace imagery, proprietary inspection images, or images containing personal information. Remove EXIF metadata from public samples where practical.

## Known limitations

CIFAR-10 is low-resolution and does not represent many real-world quality-inspection conditions. Resizing 32×32 images to 224×224 does not add information. Results may not transfer to industrial imagery, defects, lighting conditions, camera systems, or class distributions.

## License and attribution

Dataset usage must follow the original dataset terms. This repository does not redistribute the full dataset.
