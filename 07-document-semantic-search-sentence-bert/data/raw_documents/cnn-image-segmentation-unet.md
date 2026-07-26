---
project_name: Medical Image Segmentation with U-Net
project_category: CNN / Computer Vision
document_type: project_readme
tags: cnn, unet, image-segmentation, medical-imaging, dice, iou
url: https://github.com/unit-mole/cnn-projects/tree/main/01-image-segmentation-unet-medical-imaging
---
# Medical Image Segmentation with U-Net

## Objective
Segment target regions in medical images using a U-Net convolutional neural network. The pipeline includes image-mask pairing, augmentation, normalization, training callbacks, and visual comparison of predicted and reference masks.

## Metrics
Evaluation uses Dice coefficient, Intersection over Union, precision, recall, and per-image qualitative analysis. The project explains why pixel accuracy alone can be misleading for small structures.

## Deployment
A lightweight browser demonstration can display pre-generated predictions or use a converted browser model when practical. It is an educational project and not a clinical diagnostic system.
