# setup.ps1
# One-time environment setup for AMS 560 GPU Energy Control project.
# Run from Administrator PowerShell.

Write-Host "AMS 560 GPU Energy Control — Environment Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# ── Step 1: Verify NVIDIA GPU / Driver ─────────────────────────────────────
Write-Host "`n[1] Verifying NVIDIA driver ..." -ForegroundColor Yellow
nvidia-smi
if ($LASTEXITCODE -ne 0) {
    Write-Error "nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
}

# ── Step 2: Install PyTorch (CUDA 12.6, Windows, Pip, Python) ───────────────
Write-Host "`n[2] Installing PyTorch with CUDA 12.6 ..." -ForegroundColor Yellow
Write-Host "    (Stable 2.14.0 | Windows | Pip | CUDA 12.6)" -ForegroundColor Gray
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyTorch install failed. Check your internet connection and CUDA version."
    exit 1
}

# ── Step 3: Install other Python dependencies ────────────────────────────────
Write-Host "`n[3] Installing other project dependencies ..." -ForegroundColor Yellow
pip install transformers accelerate nvidia-ml-py numpy pandas matplotlib pyarrow
if ($LASTEXITCODE -ne 0) {
    Write-Error "Dependency install failed."
    exit 1
}

# ── Step 3b: Verify CUDA is accessible ───────────────────────────────────────
Write-Host "`n[3b] Verifying PyTorch CUDA ..." -ForegroundColor Yellow
python -c "import torch; ok=torch.cuda.is_available(); print('CUDA:', ok); print('GPU:', torch.cuda.get_device_name(0) if ok else 'N/A'); assert ok, 'CUDA not available'"
if ($LASTEXITCODE -ne 0) {
    Write-Error "CUDA not available after install. Ensure your NVIDIA driver supports CUDA 12.6."
    exit 1
}

# ── Step 4: Verify nvidia-smi clock control ─────────────────────────────────
Write-Host "`n[4] Testing nvidia-smi clock lock (requires Administrator) ..." -ForegroundColor Yellow
nvidia-smi -q -d SUPPORTED_CLOCKS | Select-String "Graphics" | Select-Object -First 5
$testClock = (nvidia-smi -q -d SUPPORTED_CLOCKS | Select-String "Graphics\s*:\s*(\d+)" | Select-Object -First 1) -replace ".*Graphics\s*:\s*", "" -replace "\s.*", ""
if ($testClock) {
    Write-Host "Testing clock lock to $testClock MHz ..."
    nvidia-smi -lgc "${testClock},${testClock}"
    Start-Sleep -Seconds 1
    nvidia-smi -rgc
    Write-Host "Clock control: OK" -ForegroundColor Green
} else {
    Write-Warning "Could not determine a test clock value."
}

# ── Step 5: Create results directory ────────────────────────────────────────
Write-Host "`n[5] Creating results directory ..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "results" -Force | Out-Null

# ── Step 6: Run unit tests ───────────────────────────────────────────────────
Write-Host "`n[6] Running controller unit tests ..." -ForegroundColor Yellow
python -m controllers.naive_slo
python controllers\robust_delay_budget.py

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "Setup complete. Next steps:" -ForegroundColor Green
Write-Host "  1. .\run_all.ps1 -QuickTest     # fast smoke test (5-10 min)"  -ForegroundColor White
Write-Host "  2. .\run_all.ps1                # full experiment matrix" -ForegroundColor White
Write-Host "  3. python analyze_results.py    # generate all figures" -ForegroundColor White
Write-Host "================================================" -ForegroundColor Green
