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

A 401-example self-authored and curated public-safe ML/Data Science instruction curriculum described in `DATASET_CARD.md`. Topic-grouped splits prevent prompt variants for one concept from crossing into the held-out set. The source bundle does not claim a trained adapter; run the RTX experiment notebook and update this card with actual run metadata.

## Evaluation

The executed experiment framework includes:

- held-out sequence loss and perplexity,
- category-aware instruction adherence,
- BERTScore precision, recall, and F1,
- ROUGE-1, ROUGE-2, and ROUGE-L,
- Sentence-Transformer relevance to prompts and references,
- reference-support and hallucination-risk triage,
- warm-cache latency, throughput, output length, and peak GPU memory,
- paired bootstrap confidence intervals and per-category deltas,
- before-vs-after examples,
- a stratified manual correctness, relevance, clarity, and hallucination review.

No numeric model results are claimed until `notebooks/05_end_to_end_gpu_lora_training_evaluation.ipynb` has been executed against actual saved artifacts.

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


## Reproducible experiment files

- `configs/portfolio_experiment.yaml`
- `notebooks/05_end_to_end_gpu_lora_training_evaluation.ipynb`
- `outputs/portfolio_experiment/model_metrics.json` after execution
- `outputs/portfolio_experiment/base_vs_lora_per_example.csv` after execution
- `outputs/portfolio_experiment/manual_review_results.csv` after execution

The model card should be updated with the run ID, hardware, adapter repository, exact metrics, confidence intervals, and manual-review results before public release.
