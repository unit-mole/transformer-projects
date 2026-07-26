$ErrorActionPreference = "Stop"
if (-not $env:HF_SPACE_REPO) { throw "Set HF_SPACE_REPO, e.g. anmol-unitmole/06-multimodal-visual-question-answering-transformer" }
if (-not $env:HF_TOKEN) { throw "Set HF_TOKEN to a Hugging Face write token" }

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Temp = Join-Path ([System.IO.Path]::GetTempPath()) ("hf-space-" + [guid]::NewGuid())
git clone "https://user:$($env:HF_TOKEN)@huggingface.co/spaces/$($env:HF_SPACE_REPO)" $Temp
Get-ChildItem $Temp -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force
Copy-Item (Join-Path $Root "space\*") $Temp -Recurse -Force
Push-Location $Temp
git add .
git commit -m "Deploy Project 06 static VQA Space"
if ($LASTEXITCODE -ne 0) { Write-Host "No new changes to commit." }
git push
Pop-Location
Remove-Item $Temp -Recurse -Force
