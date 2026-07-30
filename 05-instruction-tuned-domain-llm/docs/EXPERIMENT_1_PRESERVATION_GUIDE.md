# Preserve Experiment 1 Without Deleting It

## Experiment folder

Keep this folder unchanged:

```text
outputs\experiments\flan_t5_base_lora_20260730_120312\
```

It contains the first trained adapter, training metadata, evaluation outputs,
and completed human-review files. Experiment 2 uses it as a comparison baseline.

## Required notebook state

In the original Experiment 1 notebook, keep:

```python
EVALUATION_HUMAN_REVIEW_COMPLETED = True
PROMOTE_REVIEWED_ARTIFACTS = False
```

Run the release-gate cell and save the notebook. The expected message is that
promotion was skipped.

## Create the non-destructive archive

From the Project 05 folder with `.venv` activated:

```cmd
python scripts\archive_experiment_1.py
```

The script creates:

```text
outputs\experiment_archives\experiment_1_initial_lora\
├── EXPERIMENT_1_CARD.md
├── experiment_1_summary.json
├── experiment_manifest.json
├── sha256sums.txt
├── archive_result.json
├── notebook_snapshots\
│   └── 05_full_training_evaluation_pipeline.ipynb
└── experiment_1_initial_lora_full.zip
```

The original experiment directory is not moved, modified, or deleted.

## Verify the archive

```cmd
python -c "from pathlib import Path; p=Path(r'outputs\experiments\flan_t5_base_lora_20260730_120312'); a=Path(r'outputs\experiment_archives\experiment_1_initial_lora'); print('Original exists:', p.exists()); print('Archive exists:', a.exists()); print('ZIP exists:', (a/'experiment_1_initial_lora_full.zip').exists())"
```

All three values should be `True`.

## What should go to Git

Recommended to commit:

- `EXPERIMENT_1_CARD.md`
- `experiment_1_summary.json`
- reviewed comparison CSVs
- metric JSON files
- training curve
- comparison chart
- the executed notebook

Do not commit the full archive ZIP or model weights to ordinary Git history.
Keep the full ZIP locally, in protected storage, or publish an approved adapter
to a Hugging Face model repository.
