# Output Artifacts

The root of this folder contains only reviewed, promoted portfolio evidence. Full working runs are saved under `outputs/experiments/` and ignored by Git because they can contain checkpoints and raw teacher generations.

After the notebook's human-review and promotion gates, expected root artifacts include:

- `training_curve.png`
- `training_log_history.json`
- `base_model_metrics.json`
- `lora_model_metrics.json`
- `base_vs_lora_comparison.json`
- `per_example_base_vs_lora.csv`
- `before_after_finetuning_examples.md`
- `base_vs_lora_metric_comparison.png`
- `evaluation_manifest.json`
- `release_manifest.json`
- `portfolio_readiness_report.json`

Do not replace `not_run` placeholders with invented values.
