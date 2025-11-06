# Install project dependencies using China mirror

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installing Dependencies (Using China Mirror)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Install backend dependencies
Write-Host "`n[1/2] Installing backend dependencies..." -ForegroundColor Yellow
cd backend
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
if ($LASTEXITCODE -eq 0) {
    Write-Host "Backend dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "Failed to install backend dependencies" -ForegroundColor Red
    Write-Host "Trying alternative mirror..." -ForegroundColor Yellow
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
}

# Install frontend dependencies
Write-Host "`n[2/2] Installing frontend dependencies..." -ForegroundColor Yellow
cd ../frontend
npm install --registry=https://registry.npmmirror.com
if ($LASTEXITCODE -eq 0) {
    Write-Host "Frontend dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "Failed to install frontend dependencies" -ForegroundColor Red
    exit 1
}

cd ..

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nNext step: Run .\start_all.ps1 to start the system" -ForegroundColor Yellow
Write-Host "`nPress any key to exit..." -ForegroundColor Gray
Read-Host
