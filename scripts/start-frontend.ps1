# Start Vite frontend on http://localhost:5173
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\frontend"

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Node.js/npm not found. Install LTS from https://nodejs.org/ then reopen PowerShell."
    exit 1
}

if (-not (Test-Path ".\node_modules")) {
    Write-Host "Installing npm packages (first run may take a few minutes)..."
    npm install
}

Write-Host ""
Write-Host "Starting UI — open the Local URL shown below (usually http://localhost:5173)"
Write-Host "Keep this window OPEN. Press Ctrl+C to stop."
Write-Host ""

npm run dev
