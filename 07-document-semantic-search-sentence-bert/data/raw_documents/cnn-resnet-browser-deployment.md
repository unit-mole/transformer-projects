---
project_name: Image Classification with ResNet
project_category: CNN / Computer Vision
document_type: deployment_guide
tags: resnet, image-classification, transfer-learning, tensorflowjs, github-pages
url: https://github.com/unit-mole/cnn-projects/tree/main/04-image-classification-resnet
---
# Image Classification with ResNet

## Model
Transfer learning with ResNet provides a strong image-classification baseline. The workflow freezes pretrained layers, trains a task-specific classification head, fine-tunes selected layers, and evaluates class-level precision, recall, F1, and confusion matrices.

## Browser deployment
The static demo uses HTML, CSS, JavaScript, and a browser-compatible model format such as TensorFlow.js. GitHub Pages hosts the interface without a Python server. All model and label files use relative paths so deployment works inside a repository subfolder.
