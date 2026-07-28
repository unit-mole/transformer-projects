# Generated Evaluation Outputs

The committed files in this folder initially contain honest `not_run`
placeholders. Running
`notebooks/complete_longformer_training_evaluation_pipeline.ipynb` replaces or
extends them with actual benchmark outputs.

Expected artifacts include:

- `qasper_dataset_summary.json`
- `training_summary.json`
- `training/training_history.json`
- `baseline_comparison.csv` and `.json`
- `controlled_context_length_comparison.csv` and `.json`
- `evaluation_manifest.json`
- `EVALUATION_REPORT.md`
- one summary JSON per model
- one example-level CSV and JSONL per model
- context-length, answer-position, confidence, latency, and error-analysis files
- portfolio plots in PNG format

Do not manually type metric values into these files. Generate them from real
model runs. Raw QASPER data, processed datasets, caches, checkpoints, and model
weights belong outside Git and are excluded by `.gitignore`.
