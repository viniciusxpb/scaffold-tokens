# Scaffold Tokens - Windows Setup
# Run as Administrator: Right-click PowerShell -> Run as Administrator
# Then: Set-ExecutionPolicy Bypass -Scope Process; .\setup_windows.ps1

Write-Host "=== Scaffold Tokens - Windows Setup ===" -ForegroundColor Cyan
Write-Host ""

# --- 1. Check admin ---
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: Run this script as Administrator." -ForegroundColor Red
    Write-Host "  Right-click PowerShell -> Run as Administrator"
    exit 1
}

# --- 2. Install Chocolatey ---
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "[1/4] Installing Chocolatey..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    Write-Host "  Chocolatey installed." -ForegroundColor Green
} else {
    Write-Host "[1/4] Chocolatey already installed." -ForegroundColor Green
}

# --- 3. Install make ---
if (-not (Get-Command make -ErrorAction SilentlyContinue)) {
    Write-Host "[2/4] Installing make..." -ForegroundColor Yellow
    choco install make -y
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    Write-Host "  make installed." -ForegroundColor Green
} else {
    Write-Host "[2/4] make already installed." -ForegroundColor Green
}

# --- 4. Install micromamba ---
if (-not (Get-Command micromamba -ErrorAction SilentlyContinue)) {
    Write-Host "[3/4] Installing micromamba..." -ForegroundColor Yellow
    choco install micromamba -y
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    Write-Host "  micromamba installed." -ForegroundColor Green
} else {
    Write-Host "[3/4] micromamba already installed." -ForegroundColor Green
}

# --- 5. Replace Makefile with Windows version ---
Write-Host "[4/4] Setting up Windows Makefile..." -ForegroundColor Yellow
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$windowsMakefile = Join-Path $scriptDir "utils\Makefile.windows"
$mainMakefile = Join-Path $scriptDir "Makefile"

if (Test-Path $windowsMakefile) {
    Copy-Item $windowsMakefile $mainMakefile -Force
    Write-Host "  Makefile replaced with Windows version." -ForegroundColor Green
} else {
    Write-Host "  ERROR: utils\Makefile.windows not found." -ForegroundColor Red
    exit 1
}

# --- Done ---
Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Open a NEW terminal (so PATH updates take effect)"
Write-Host "  2. Run: make setup"
Write-Host "  3. Run: make download-model"
Write-Host "  4. Run: make generate"
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Yellow
make help
