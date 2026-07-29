# Project 03 Large-Scale Benchmark Execution Guide

## Objective

This evaluation upgrade addresses the main experimental limitations of the
portfolio project:

- replaces sample-only evidence with recognized BEIR benchmarks;
- adds TF-IDF and BM25 baselines;
- evaluates the MiniLM bi-encoder;
- evaluates MiniLM plus MS MARCO cross-encoder reranking;
- measures Recall@K, Precision@10, Hit@10, MRR@10, nDCG@10 and MAP@100;
- measures indexing, query encoding, retrieval and reranking latency;
- computes paired bootstrap confidence intervals for reranking improvement;
- saves every result to JSON, CSV, Markdown and PNG files;
- optionally fine-tunes the bi-encoder using BM25 hard negatives.

## Recommended benchmark scale

Run two official BEIR datasets:

```text
SciFact   → scientific claim and evidence retrieval
NFCorpus  → biomedical information retrieval
```

The public Static Space should remain small and fast. The larger datasets are
used offline for rigorous evidence and should not be committed to GitHub.

## Step 1 — Open PowerShell in Project 03

```powershell
cd "03-cross-encoder-bi-encoder-ranking-system"
```

## Step 2 — Create the benchmark environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_benchmark_windows.ps1
```

The script installs the project benchmark requirements, registers a Jupyter
kernel and verifies that PyTorch can see the RTX GPU.

### When CUDA is not detected

Run:

```powershell
python scripts\check_gpu.py
```

If `cuda_available` is false, reinstall PyTorch using the current command from
the official PyTorch **Start Locally** selector for:

```text
OS: Windows
Package: Pip
Language: Python
Compute platform: the newest CUDA version supported by your driver
```

Then rerun `python scripts\check_gpu.py`.

## Step 3 — Run the master notebook

```powershell
jupyter lab notebooks\04-large-scale-ranking-benchmark.ipynb
```

Select the kernel:

```text
Project 03 Ranking Benchmark
```

Execute the notebook from top to bottom.

## Step 4 — Base-model benchmark

The notebook evaluates:

```text
TF-IDF
BM25
all-MiniLM-L6-v2 bi-encoder
all-MiniLM-L6-v2 + ms-marco-MiniLM-L-6-v2 reranker
```

Default GPU-oriented settings:

```text
Candidate K: 100
Rerank K: 100
Bi-encoder batch size: 128
Cross-encoder batch size: 64
Bootstrap resamples: 2,000
```

Reduce batch sizes only if an out-of-memory error occurs.

## Step 5 — Review generated artifacts

The current run is copied to:

```text
outputs/benchmark/latest/
```

Review:

```text
benchmark_summary.csv
benchmark_summary.json
per_query_metrics.csv
bootstrap_significance.json
ranking_examples.csv
latency_breakdown.csv
metric_comparison.png
recall_at_k_curves.png
latency_comparison.png
reranking_delta_distribution.png
PORTFOLIO_RESULTS.md
```

## Step 6 — Sync verified metrics into the portfolio

After checking that the numbers are reasonable:

```powershell
python scripts\sync_benchmark_results.py
```

This creates or updates:

```text
BENCHMARK_RESULTS.md
model_hub/pipeline-card/evaluation_results.json
```

## Step 7 — Optional fine-tuning upgrade

The notebook includes an optional SciFact fine-tuning stage using:

```text
Base model: all-MiniLM-L6-v2
Training split: SciFact train
Negatives: BM25 hard negatives
Loss: MultipleNegativesRankingLoss
GPU precision: BF16 when supported, otherwise FP16
```

Command-line alternative:

```powershell
python scripts\fine_tune_bi_encoder.py --device cuda --epochs 2 --batch-size 32
```

The model is saved locally under:

```text
models/fine_tuned/docrank360-minilm-scifact/
```

The directory is ignored by Git because model weights belong on Hugging Face
Model Hub rather than inside the GitHub repository.

## Step 8 — Evaluate the fine-tuned model

```powershell
python scripts\run_portfolio_benchmark.py `
  --datasets scifact nfcorpus `
  --bi-encoder-model "models/fine_tuned/docrank360-minilm-scifact" `
  --model-label "fine_tuned_scifact" `
  --device cuda `
  --candidate-k 100 `
  --rerank-k 100
```

Evaluate both SciFact and NFCorpus. SciFact measures in-domain improvement;
NFCorpus shows whether the fine-tuning generalizes or damages transfer quality.

## Step 9 — Manual error analysis

Use `ranking_examples.csv` to record:

- largest reranking improvements;
- largest reranking regressions;
- missed relevant documents;
- lexical baseline wins;
- ambiguous scientific or biomedical claims;
- cases where cross-encoder confidence appears misleading.

Update:

```text
outputs/manual_relevance_analysis.md
```

## Step 10 — Push only the project and workflow

```bash
git add "03-cross-encoder-bi-encoder-ranking-system" ".github/workflows/03-cross-encoder-bi-encoder-ranking-system.yml"
git commit -m "Upgrade Project 03 with large-scale BEIR evaluation and GPU benchmarking"
git push origin main
```
