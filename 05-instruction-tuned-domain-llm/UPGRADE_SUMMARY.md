# Project 05 — Portfolio 9/10 Upgrade Summary

## What was upgraded

- Changed the quality preset from FLAN-T5-small to `google/flan-t5-base`.
- Added automatic RTX/CUDA/VRAM/BF16 detection and hardware-aware batch settings.
- Increased LoRA capacity to rank 16 / alpha 32 and added cosine scheduling, label smoothing, mixed precision, gradient checkpointing, early stopping, best-checkpoint restoration, test loss, and validation perplexity.
- Added a local teacher-model dataset expansion workflow targeting approximately 600 records.
- Added near-duplicate removal, benchmark-leakage screening, stratified splits, generation retries, audit logs, and mandatory human review.
- Added an independent 80-example self-authored benchmark with reference answers.
- Added paired base-versus-LoRA evaluation using BERTScore, ROUGE-L, sentence-embedding similarity, TF-IDF relevance, instruction adherence, response-quality rubric, latency, and hallucination-risk flags.
- Added category/difficulty slices, per-example deltas, win rates, and 95% bootstrap confidence intervals.
- Added manual factuality, relevance, clarity, instruction-following, hallucination, and preferred-model review fields.
- Added release promotion with SHA-256 checksums.
- Added an evidence-based portfolio-readiness checker with a 9/10 target.
- Added the full notebook `notebooks/05_full_training_evaluation_pipeline.ipynb`.
- Added Windows RTX setup, training, evaluation, Git, and Hugging Face deployment guidance.
- Updated the Gradio app, README, model card, dataset card, CI workflow, tests, and project documentation.

## Current honest status

The upgraded pipeline is complete and tested, but the local RTX experiment has not been executed in this package. Therefore:

- `ml_ds_instruction_dataset_v2.jsonl` has not been generated;
- LoRA adapter weights do not yet exist;
- real training curves and numeric model metrics do not yet exist;
- human review and Hugging Face deployment have not yet occurred.

Run the full notebook locally, review the artifacts, and promote the experiment before presenting the project as completed.

## Primary execution file

```text
notebooks/05_full_training_evaluation_pipeline.ipynb
```

## Primary guide

```text
docs/LOCAL_RTX_FULL_EXPERIMENT_GUIDE.md
```
