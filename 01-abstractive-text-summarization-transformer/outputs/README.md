# Output Artifacts

Committed files are schemas or `not_run` placeholders. Run evaluation to create actual outputs under `outputs/runs/<timestamp>/`.

Expected artifacts:

- `model_metrics.json`
- `rouge_scores.json`
- `bertscore_results.json`
- `inference_time_results.json`
- `generated_summary_examples.csv`
- `transformer_vs_lstm_comparison.csv`
- `transformer_vs_lstm_comparison.png`
- `error_analysis_examples.md`

Never replace placeholders with invented metrics.
