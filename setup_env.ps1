# Phase 1 setup for Windows. Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }   # then put your key in .env
.\.venv\Scripts\python.exe phase1_check_env.py
Write-Host "Activate later with: .\.venv\Scripts\Activate.ps1"
