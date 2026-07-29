# Start Here — Upgrade Project 03 to a 9/10 Portfolio Project

## Phase A — Required experimental evidence

1. Extract the update into the existing `transformer-projects` folder.
2. Open PowerShell inside `03-cross-encoder-bi-encoder-ranking-system`.
3. Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_benchmark_windows.ps1
```

4. Confirm the RTX GPU:

```powershell
python scripts\check_gpu.py
```

5. Launch the master notebook:

```powershell
jupyter lab notebooks\04-large-scale-ranking-benchmark.ipynb
```

6. Execute the notebook through the **base-model benchmark** and review:

```text
outputs/benchmark/latest/
```

7. Sync verified values:

```powershell
python scripts\sync_benchmark_results.py
```

8. Complete manual error analysis in:

```text
outputs/manual_relevance_analysis.md
```

## Phase B — Optional fine-tuning

Run only after the base benchmark is complete:

```powershell
python scripts\fine_tune_bi_encoder.py --device cuda --epochs 2 --batch-size 32
```

Then evaluate the saved model on both SciFact and NFCorpus. Publish a personal
model repository only when the held-out evaluation supports the claim.

## Phase C — Push the reviewed results

```bash
git add "03-cross-encoder-bi-encoder-ranking-system" ".github/workflows/03-cross-encoder-bi-encoder-ranking-system.yml"
git commit -m "Upgrade Project 03 with large-scale BEIR evaluation and GPU benchmarking"
git push origin main
```
