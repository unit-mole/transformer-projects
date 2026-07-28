@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python deploy_static_space.py
if errorlevel 1 (
  echo.
  echo Deployment failed. Make sure you are logged in with: hf auth login
  pause
  exit /b 1
)
echo.
echo Open the final Space:
echo https://huggingface.co/spaces/anmol-unitmole/long-document-question-answering-longformer
pause
