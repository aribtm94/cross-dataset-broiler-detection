<#
.SYNOPSIS
    Create the two Python 3.10 virtual environments this project needs
    (.venv-yolo and .venv-mowa) with CUDA 12.1 torch, then install deps.

.DESCRIPTION
    Reproduces the exact environments used for the thesis experiments.
    A CUDA 12.1-capable NVIDIA GPU is required (MOWA hard-codes .cuda()).

.EXAMPLE
    ./scripts/setup_env.ps1
    ./scripts/setup_env.ps1 -Python "C:\Python310\python.exe"
#>
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$TorchIndex = "https://download.pytorch.org/whl/cu121"
$TorchPkgs  = @("torch==2.1.2", "torchvision==0.16.2")

function New-Venv([string]$Name, [string]$Requirements) {
    Write-Host "`n=== Creating $Name ===" -ForegroundColor Cyan
    & $Python -m venv $Name
    $py = Join-Path $Root "$Name\Scripts\python.exe"
    & $py -m pip install --upgrade pip
    Write-Host "--- Installing CUDA torch (cu121) first ---" -ForegroundColor Yellow
    & $py -m pip install @TorchPkgs --index-url $TorchIndex
    Write-Host "--- Installing $Requirements ---" -ForegroundColor Yellow
    & $py -m pip install -r $Requirements
    Write-Host "$Name ready." -ForegroundColor Green
}

# Sanity check: python version
$ver = & $Python --version 2>&1
Write-Host "Using interpreter: $ver"
if ($ver -notmatch "3\.10") {
    Write-Warning "Python 3.10 is recommended; you are on '$ver'. Continuing anyway."
}

New-Venv ".venv-yolo" "requirements-yolo.txt"
New-Venv ".venv-mowa" "requirements-mowa.txt"

Write-Host "`nBoth environments created." -ForegroundColor Green
Write-Host "Next: clone MOWA + download assets. See docs/DATA_SETUP.md." -ForegroundColor Green
