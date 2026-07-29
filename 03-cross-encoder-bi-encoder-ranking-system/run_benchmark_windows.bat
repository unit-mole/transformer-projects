@echo off
call .venv-benchmark\Scripts\activate.bat
python scripts\check_gpu.py
python scripts\run_portfolio_benchmark.py --datasets scifact nfcorpus --device cuda --candidate-k 100 --rerank-k 100 --bi-batch-size 128 --cross-batch-size 64 --bootstrap-samples 2000
python scripts\sync_benchmark_results.py
pause
