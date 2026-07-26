$ErrorActionPreference = "Stop"

Write-Host "Creating Python environment..."
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

Write-Host "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Installing browser dependencies..."
Push-Location web
npm install
Pop-Location

Write-Host ""
Write-Host "Setup complete."
Write-Host "Python Gradio demo: python app.py"
Write-Host "Static browser demo: cd web; npm run dev"
