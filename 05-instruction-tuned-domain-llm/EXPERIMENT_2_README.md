# Project 05 Experiment 2 Upgrade

This overlay adds a non-destructive Experiment 1 archive workflow and a complete
Experiment 2 pipeline built around curated Version 3 training data.

## Important extraction rule

Merge these files into your existing:

```text
transformer-projects\05-instruction-tuned-domain-llm\
```

Do not delete or replace the existing `outputs` directory. It contains
Experiment 1.

## Start here

1. Read `docs/EXPERIMENT_1_PRESERVATION_GUIDE.md`.
2. Run `python scripts\archive_experiment_1.py`.
3. Run `python -m pytest -q`.
4. Open `notebooks\05_experiment_2_quality_upgrade_pipeline.ipynb`.
5. Run sequentially and stop at each human-review gate.

## Experiment design

Experiment 1 remains the honest baseline. Experiment 2 changes the dataset
quality, adapter capacity, learning rate, warmup implementation, seed, and
release comparison while keeping the base model and held-out benchmark fixed.
