$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment..."
python -m venv .venv

Write-Host "Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing project dependencies..."
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "Run the app with: python app.py"
