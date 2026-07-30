# Data Directory

This directory separates seed training data, locally generated training data, and the held-out benchmark.

## Files

- `ml_ds_instruction_dataset.jsonl` — 93 self-authored seed records from the original portfolio scaffold.
- `dataset_generation_plan.json` — 64 ML/Data Science topics used by the local teacher-model expansion workflow.
- `ml_ds_instruction_dataset_v2.jsonl` — generated locally by the full experiment notebook; do not treat it as reviewed until the human approval gate is completed.
- `benchmark_prompts_v2.jsonl` — 80 self-authored held-out prompts with reference answers; never use this file for training.
- `sample_instructions.jsonl` — small demo prompts used by the Gradio interface.
- `evaluation_prompts.jsonl` — legacy small evaluation file retained for backward compatibility.

## Recommended Workflow

Run `notebooks/05_full_training_evaluation_pipeline.ipynb` or:

```bash
python scripts/generate_enhanced_dataset.py --target-examples 600
```

The generation stage saves raw teacher outputs and audit files under `outputs/<experiment>/dataset_generation/`. Review a stratified sample, all advanced records, and all code records before setting the notebook approval flag.

## Public-Data Rules

Do not include private company records, customer identifiers, confidential quality cases, personal information, proprietary course material, or copyrighted textbook passages. Use generic and non-confidential examples.
