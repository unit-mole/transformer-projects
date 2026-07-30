# Experiment 1 — Initial FLAN-T5-base LoRA Run

## Status

**Preserved, reviewed, and intentionally not promoted.**

## Purpose

This run established the first complete training and evaluation baseline for the
ML/Data Science Learning Assistant. It is retained as evidence of an honest
iteration cycle rather than deleted after weak response-quality findings.

## Core results

- Base model: `google/flan-t5-base`
- Fine-tuning: `LoRA/PEFT`
- Validation loss: `3.375509738922119`
- Test loss: `3.424408435821533`
- Validation perplexity: `29.239184`
- Trainable percentage: `0.709641`
- LoRA preferred by human review: `51`
- Base preferred by human review: `18`
- Ties: `11`
- LoRA mean factuality: `1.4125` / 5
- LoRA mean relevance: `2.4` / 5
- LoRA mean clarity: `2.3125` / 5
- LoRA mean instruction following: `1.475` / 5

## Decision

The adapter was not promoted because human evaluation found technically weak,
circular, incomplete, or hallucinated answers. Experiment 2 therefore changes
the supervision quality and keeps the held-out benchmark unchanged.

## Recruiter-facing value

This experiment demonstrates reproducible GPU training, LoRA/PEFT, held-out
benchmarking, automated metrics, human review, release gating, and evidence-based
iteration rather than selective reporting.
