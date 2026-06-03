# CLARYON one-shot installer (Windows / PowerShell).
#
# Usage:
#     .\scripts\install.ps1                  # full install with pyradiomics
#     .\scripts\install.ps1 -NoRadiomics     # skip pyradiomics
#     $env:PYTHON = 'C:\Python312\python.exe'; .\scripts\install.ps1
#
# Creates .venv in the project root (if not present), installs CLARYON in
# editable mode with all standard extras, and handles the pyradiomics
# PEP 517 build-isolation workaround.
#
# If you get a script-execution error, allow local scripts for this session:
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

[CmdletBinding()]
param(
    [switch]$NoRadiomics
)

$ErrorActionPreference = 'Stop'

# --- Locate repo root (one level up from scripts/) ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path
Set-Location $RepoRoot

if (-not (Test-Path 'pyproject.toml')) {
    Write-Error "pyproject.toml not found. Run from the CLARYON repo root."
    exit 1
}

$installRadiomics = -not $NoRadiomics

Write-Host "=========================================="
Write-Host " CLARYON one-shot installer (Windows)"
Write-Host " Repo root : $RepoRoot"
Write-Host " Radiomics : $(if ($installRadiomics) { 'yes' } else { 'skipped' })"
Write-Host "=========================================="

# --- Stage 1: Python version check ---
Write-Host ""
Write-Host "[1/5] Checking Python version..."
$pythonExe = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
try {
    $pyVer = & $pythonExe -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    $pyOk  = & $pythonExe -c 'import sys; print(1 if sys.version_info >= (3, 11) else 0)'
}
catch {
    Write-Error "'$pythonExe' not found on PATH. Install Python 3.11+ first, or set `$env:PYTHON to a 3.11+ interpreter."
    exit 1
}
if ($pyOk.Trim() -ne '1') {
    Write-Error "Python $pyVer detected. CLARYON requires Python >= 3.11. Set `$env:PYTHON to a 3.11+ interpreter and re-run."
    exit 1
}
Write-Host "      Python $pyVer OK."

# --- Stage 2: Virtual env ---
Write-Host ""
Write-Host "[2/5] Setting up virtual environment (.venv)..."
if (Test-Path '.venv') {
    Write-Host "      .venv already exists - reusing."
}
else {
    & $pythonExe -m venv .venv
    Write-Host "      Created .venv."
}
$venvPy  = Join-Path '.venv' 'Scripts\python.exe'
$venvPip = Join-Path '.venv' 'Scripts\pip.exe'

if (-not (Test-Path $venvPy)) {
    Write-Error "$venvPy not found after venv creation."
    exit 1
}

# --- Stage 3: Build prereqs ---
Write-Host ""
Write-Host "[3/5] Upgrading pip / setuptools / wheel..."
& $venvPip install --upgrade pip setuptools wheel

# --- Stage 4: CLARYON + standard extras ---
Write-Host ""
Write-Host "[4/5] Installing CLARYON in editable mode with [all] extras..."
Write-Host "      (PennyLane, PyTorch, XGBoost, LightGBM, CatBoost, TabPFN, SHAP, LIME, ...)"
& $venvPip install -e ".[all]"

# --- Stage 5: pyradiomics (optional, with build-isolation workaround) ---
Write-Host ""
if ($installRadiomics) {
    Write-Host "[5/5] Installing pyradiomics (with --no-build-isolation workaround)..."
    Write-Host "      pyradiomics 3.x's setup.py imports numpy at the top level, so the"
    Write-Host "      isolated PEP 517 build env fails. We pre-install numpy + versioneer,"
    Write-Host "      then install pyradiomics with build isolation disabled."
    & $venvPip install numpy versioneer
    & $venvPip install -e ".[radiomics]" --no-build-isolation
}
else {
    Write-Host "[5/5] Skipping pyradiomics install (-NoRadiomics)."
}

# --- Smoke test ---
Write-Host ""
Write-Host "=========================================="
Write-Host " Smoke test: importing claryon..."
Write-Host "=========================================="
& $venvPy -c "import claryon; print(f'CLARYON {claryon.__version__} imported OK')"

Write-Host ""
Write-Host "=========================================="
Write-Host " Install complete."
Write-Host ""
Write-Host " Activate the env:    .\.venv\Scripts\Activate.ps1"
Write-Host " Verify registry:     claryon list-models"
Write-Host " Run an example:      claryon run -c configs\example_tabular.yaml"
Write-Host "=========================================="
