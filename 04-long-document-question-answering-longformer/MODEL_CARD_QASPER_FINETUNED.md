---
language: en
license: mit
library_name: transformers
pipeline_tag: question-answering
base_model: valhalla/longformer-base-4096-finetuned-squadv1
datasets:
  - allenai/qasper
tags:
  - longformer
  - question-answering
  - long-document-qa
  - qasper
  - document-ai
---

# Longformer QASPER Extractive Document QA

This model card is the upload template for the checkpoint produced by
`notebooks/complete_longformer_training_evaluation_pipeline.ipynb`.

## Base model

`valhalla/longformer-base-4096-finetuned-squadv1`

## Project fine-tuning

The checkpoint is further fine-tuned by this project on a documented subset of
QASPER v0.3 containing only answerable questions with one contiguous extractive
span that can be located in the reconstructed paper text. Free-form, yes/no,
unanswerable, unresolved, and multi-span annotations are excluded because this
model predicts one contiguous span.

## Evaluation

The model was evaluated on 200 contiguous-extractive examples from the
QASPER validation set.

| Model | Exact Match | Token F1 | Evidence Recovery | Evidence Token Recall |
|---|---:|---:|---:|---:|
| BERT truncated to 512 tokens | 1.50% | 7.37% | 26.00% | 41.34% |
| Base Longformer with sliding windows | 6.00% | 16.16% | 30.00% | 45.88% |
| QASPER-fine-tuned Longformer | **12.50%** | **26.66%** | **49.00%** | **60.14%** |

### Training configuration

- Training examples: 803
- Validation examples: 419
- Evaluation examples: 200
- Maximum training length: 3,072 tokens
- Sliding-window stride: 384 tokens
- Training epochs: 2
- Learning rate: 1e-5
- Precision: BF16
- GPU: NVIDIA GeForce RTX 5090
- Fine-tuned by this project: Yes

### Performance interpretation

The QASPER-fine-tuned Longformer outperformed both the truncated BERT baseline
and the original SQuAD-fine-tuned Longformer checkpoint across Exact Match,
token-level F1, evidence recovery, and evidence-token recall.

The model remains imperfect. Exact Match is a strict metric, and QASPER
questions frequently contain long, technical, and semantically complex answers.
The reported values should not be interpreted as production-level reliability.