# ATC Speech Recognition System - One-Click Startup Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ATC Speech Recognition System Starting..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check Python environment
Write-Host "`n[1/4] Checking Python environment..." -ForegroundColor Yellow
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonVersion = python --version
    Write-Host "OK - $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "ERROR - Python not installed" -ForegroundColor Red
    exit 1
}

# Check Node.js environment
Write-Host "`n[2/4] Checking Node.js environment..." -ForegroundColor Yellow
if (Get-Command node -ErrorAction SilentlyContinue) {
    $nodeVersion = node --version
    Write-Host "OK - Node.js $nodeVersion" -ForegroundColor Green
} else {
    Write-Host "ERROR - Node.js not installed" -ForegroundColor Red
    exit 1
}

# Start backend service (with conda activation)
Write-Host "`n[3/4] Starting backend service..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; conda activate Whisper; Write-Host 'Backend service starting...' -ForegroundColor Cyan; python backend/app.py"
Write-Host "OK - Backend service started in new window (http://localhost:8000)" -ForegroundColor Green

# Wait for backend to start
Start-Sleep -Seconds 5

# Start frontend service
Write-Host "`n[4/4] Starting frontend service..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend'; Write-Host 'Frontend service starting...' -ForegroundColor Cyan; npm start"
Write-Host "OK - Frontend service started in new window (http://localhost:3000)" -ForegroundColor Green

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "System Started Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nAccess URLs:" -ForegroundColor Yellow
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "`nPress any key to exit..." -ForegroundColor Gray
Read-Host
