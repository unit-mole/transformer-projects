# Experiment 2 — Curated Dataset V3 and Second LoRA Run

## Goal

Experiment 2 tests a specific hypothesis:

> Improving supervision quality will improve factuality and instruction
> following more effectively than simply training Experiment 1 for more epochs.

The held-out 80-prompt benchmark remains unchanged.

## New files

```text
data\
├── curated_topic_cards_v3.json
├── curated_comparisons_v3.json
├── curated_code_examples_v3.json
├── curated_workflows_v3.json
└── experiment2_quality_rules.json

src\
├── experiment_archive.py
├── experiment2_dataset.py
├── experiment2_training.py
└── experiment2_comparison.py

scripts\
├── archive_experiment_1.py
├── build_dataset_v3.py
└── compare_experiment_1_vs_2.py

notebooks\
└── 05_experiment_2_quality_upgrade_pipeline.ipynb
```

## Why Version 3 is different

Version 3 defaults to:

- retaining the original self-authored seed records;
- excluding Experiment 1 teacher-generated records;
- adding 275 reviewed topic-card records;
- adding 40 comparison records;
- adding 20 code records;
- adding 15 workflow and governance records;
- screening prompts against the unchanged benchmark;
- checking weak phrases, circular definitions, duplicates, and output length;
- creating deterministic train, validation, and test splits.

The expected final dataset contains roughly 440 records. It is smaller than a
randomly expanded dataset by design, but the supervision is substantially more
controlled.

## Installation check

The existing RTX environment is sufficient. Install only the test dependency if
needed:

```cmd
python -m pip install -r requirements-experiment2.txt
python -m pytest -q
```

## Run order

1. Open JupyterLab from the Project 05 folder.
2. Open `notebooks\05_experiment_2_quality_upgrade_pipeline.ipynb`.
3. Select the RTX 5090 kernel.
4. Run cells in order.
5. Preserve Experiment 1 when prompted.
6. Build Version 3.
7. Review `dataset_v3_review_sample.csv`.
8. Set `DATASET_V3_HUMAN_REVIEW_APPROVED = True` only after review.
9. Train Experiment 2.
10. Evaluate Base versus Experiment 2.
11. Compare Experiment 1 versus Experiment 2.
12. Complete all human-review CSV files.
13. Set `EXPERIMENT2_HUMAN_REVIEW_COMPLETED = True`.
14. Run the release-quality assessment.
15. Promote only when `ready` is `True`.

## Experiment 2 training differences

- LoRA rank: 32
- LoRA alpha: 64
- Learning rate: 5e-5
- Maximum epochs: 5
- Warmup: 20 steps
- Seed: 52
- Early stopping: 2 validation evaluations
- BF16 on RTX 5090

The training module also corrects the Experiment 1 warnings by using
`warmup_steps`, setting `tie_word_embeddings=False` before loading, and removing
the early-stopping callback before post-training validation and test evaluation.

## Human review files

Complete these after evaluation:

```text
outputs\experiments\<experiment-2-run>\evaluation\lora_model\manual_review_results.csv
outputs\experiments\<experiment-2-run>\evaluation\comparison\per_example_base_vs_lora.csv
outputs\experiments\<experiment-2-run>\evaluation\experiment1_vs_experiment2\experiment1_vs_experiment2_per_example.csv
```

Use `experiment1`, `experiment2`, or `tie` in the final comparison file.

## Release targets

The automatic gate requires:

- every Experiment 2 human rating completed;
- every Experiment 1 versus 2 preference completed;
- Experiment 2 preferred on at least 60 percent of reviewed prompts;
- mean factuality, relevance, clarity, and instruction following of at least 4/5;
- human hallucination rate below 10 percent.

These are portfolio release targets, not universal scientific thresholds.
