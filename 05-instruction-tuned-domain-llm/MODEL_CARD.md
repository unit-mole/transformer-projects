# Model Card — ML/DS Instruction-Tuned FLAN-T5 LoRA Adapter

## Model description

- **Base model:** `google/flan-t5-small`
- **Architecture:** encoder-decoder Transformer
- **Task:** sequence-to-sequence instruction following
- **Fine-tuning:** LoRA through Hugging Face PEFT
- **PEFT task type:** `SEQ_2_SEQ_LM`
- **Target modules:** `q`, `v`
- **Default LoRA configuration:** rank 8, alpha 16, dropout 0.05

## Intended use

Educational explanations, algorithm comparisons, metric explanations, small code examples, ML workflow guidance, interview preparation, and quality analytics learning scenarios.

## Not intended for

Legal, medical, financial, immigration, safety-critical, official, or autonomous decision-making. It is not a replacement for a domain expert, trusted documentation, or code execution and review.

## Training data

A self-authored public ML/Data Science instruction curriculum described in `DATASET_CARD.md`. The generated bundle does not contain a trained adapter; run the training workflow and update this card with actual run metadata.

## Evaluation

Planned evaluation includes:

- instruction adherence rubric,
- BERTScore precision, recall, and F1 where references exist,
- heuristic response relevance,
- latency,
- manual review,
- before-vs-after base model comparison,
- hallucination and unsupported-claim analysis.

No numeric model results are claimed until `scripts/evaluate_model.py` has been run against actual saved artifacts.

## Limitations and risks

The small model and compact dataset can produce incomplete explanations, incorrect code, outdated statements, bias, and hallucinations. Heuristic scores are not factuality guarantees. CPU inference may be slow on initial load.

## Deployment

The Gradio app loads the base model and, when configured, a PEFT adapter from the Hugging Face Hub. Training is never performed during app startup.

## Example input

```text
Instruction: Explain precision vs recall with a quality analytics example.
Response:
```

## Responsible use

All generated explanations, code, and recommendations require human review. Do not submit private or sensitive content to the public Space.
