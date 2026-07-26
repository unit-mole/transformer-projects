# Generated Outputs

The repository does **not** claim unexecuted Transformer metrics.

Run:

```bash
python scripts/build_index.py
python scripts/evaluate_model.py
python scripts/benchmark_latency.py
```

The scripts replace the `status: not_run` JSON placeholders with measured values
and generate:

- `ranking_examples.csv`
- `retrieval_recall_at_k.json`
- `mrr_at_10.json`
- `ndcg_at_10.json`
- `model_metrics.json`
- `latency_results.csv`
- `latency_results.json`
- `bi_encoder_vs_cross_encoder_comparison.png`
- `latency_by_top_k.png`

Model download and first-run index construction are excluded from warm per-query
latency summaries, but index build time is reported separately.
