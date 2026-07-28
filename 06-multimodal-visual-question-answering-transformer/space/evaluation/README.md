# Synthetic 60-pair VQA evaluation suite

This folder contains a deterministic, balanced portfolio evaluation dataset:

- 60 image-question pairs;
- 10 records each for color, object, counting, yes/no, action or scene, and spatial relationships;
- safe project-generated PNG images;
- accepted-answer lists for normalized browser scoring.

The dataset is intended for portfolio diagnostics and regression testing. It is
not an official VQA v2 benchmark and is not representative of all real-world
images.

Regenerate or validate it with:

```bash
python scripts/generate_synthetic_evaluation_set.py
python scripts/generate_synthetic_evaluation_set.py --check
```
