# Local Run Guide

## Windows PowerShell

```powershell
cd path\to\transformer-projects\01-abstractive-text-summarization-transformer
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860` if the browser does not open automatically.

## Tests

```powershell
pip install -r requirements-dev.txt
$env:SKIP_MODEL_LOAD="1"
pytest -q
Remove-Item Env:SKIP_MODEL_LOAD
```

## Evaluation

```powershell
python scripts/evaluate_model.py --input-csv data/sample_summaries.csv --compute-bertscore
```

The first model run downloads and caches the base checkpoint. Allow time and sufficient disk space. No training happens when the app starts.
