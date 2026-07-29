$ErrorActionPreference = "Stop"

Write-Host "Creating the Project 03 benchmark environment..."
python -m venv .venv-benchmark
& .\.venv-benchmark\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-benchmark.txt

Write-Host "Registering the Jupyter kernel..."
python -m ipykernel install --user --name project03-ranking-benchmark --display-name "Project 03 Ranking Benchmark"

Write-Host "Checking CUDA and RTX GPU access..."
python scripts\check_gpu.py

Write-Host ""
Write-Host "Setup completed."
Write-Host "Open the notebook with: jupyter lab notebooks\04-large-scale-ranking-benchmark.ipynb"
