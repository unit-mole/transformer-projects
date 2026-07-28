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

Copy the actual generated values from:

- `outputs/baseline_comparison.json`
- `outputs/longformer_qasper_fine-tuned_summary.json`
- `outputs/controlled_context_length_comparison.json`
- `outputs/EVALUATION_REPORT.md`

Do not publish placeholder values or the original base-model SQuAD metrics as
results produced by this project.

## Intended use

Educational and portfolio demonstrations of extractive question answering over
scientific papers, reports, quality documents, SOPs, CAPA records, supplier
reports, technical manuals, and related long documents.

## Limitations

- The model predicts extractive spans and cannot reliably answer questions that
  require free-form synthesis, yes/no reasoning, or unsupported inference.
- QASPER contains scientific NLP papers and does not directly represent every
  quality or business-document domain.
- The confidence value shown by the application is an uncalibrated proxy.
- Answers and highlighted evidence require human review.

## Responsible use

Do not use the model as the sole basis for medical, legal, financial,
safety-critical, regulatory, academic, official, or business-critical decisions.
Do not submit private or confidential documents to a public Space.
