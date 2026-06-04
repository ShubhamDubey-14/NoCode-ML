# Start FastAPI backend on http://127.0.0.1:8000
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\backend"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
}

Write-Host "Installing Python dependencies (first run may take a few minutes)..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Starting API at http://127.0.0.1:8000"
Write-Host "Test in browser: http://127.0.0.1:8000/api/health"
Write-Host "Keep this window OPEN. Press Ctrl+C to stop."
Write-Host ""

.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
