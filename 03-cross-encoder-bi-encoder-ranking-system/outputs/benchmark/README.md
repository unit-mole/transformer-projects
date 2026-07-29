# Large-Scale Benchmark Outputs

Run the master notebook or CLI to generate actual results:

```bash
python scripts/run_portfolio_benchmark.py \
  --datasets scifact nfcorpus \
  --candidate-k 100 \
  --rerank-k 100 \
  --device cuda
```

Each run creates a timestamped folder and refreshes `outputs/benchmark/latest/`.
Generated artifacts include:

- `benchmark_summary.csv` and `.json`
- `per_query_metrics.csv`
- `latency_breakdown.csv` and `.json`
- `bootstrap_significance.json`
- `dataset_metadata.json`
- `ranking_examples.csv`
- `reranking_deltas.csv`
- `metric_comparison.png`
- `recall_at_k_curves.png`
- `latency_comparison.png`
- `reranking_delta_distribution.png`
- `PORTFOLIO_RESULTS.md`

After reviewing the results, run:

```bash
python scripts/sync_benchmark_results.py
```

This copies the verified results into the Model Hub pipeline card and creates
`BENCHMARK_RESULTS.md` for GitHub.
