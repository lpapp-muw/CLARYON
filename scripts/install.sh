#!/usr/bin/env bash
# CLARYON one-shot installer (Linux / macOS).
#
# Usage:
#     bash scripts/install.sh                              # full install with pyradiomics + GPU torch
#     bash scripts/install.sh --no-radiomics               # skip pyradiomics
#     bash scripts/install.sh --cpu-only                   # CPU-only torch (saves ~3GB)
#     bash scripts/install.sh --no-radiomics --cpu-only    # both flags
#     PYTHON=/path/to/python3.12 bash scripts/install.sh
#
# Creates .venv in the project root (if not present), installs CLARYON in
# editable mode with all standard extras, and handles the pyradiomics
# PEP 517 build-isolation workaround.

set -euo pipefail

# --- Parse args ---
INSTALL_RADIOMICS=1
CPU_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --no-radiomics) INSTALL_RADIOMICS=0 ;;
        --cpu-only)     CPU_ONLY=1 ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Usage: bash scripts/install.sh [--no-radiomics] [--cpu-only]" >&2
            exit 2 ;;
    esac
done

# --- Locate repo root (one level up from scripts/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f pyproject.toml ]]; then
    echo "ERROR: pyproject.toml not found. Run from the CLARYON repo root." >&2
    exit 1
fi

echo "=========================================="
echo " CLARYON one-shot installer"
echo " Repo root : $REPO_ROOT"
echo " Radiomics : $([[ $INSTALL_RADIOMICS == 1 ]] && echo "yes" || echo "skipped")"
echo "=========================================="

# --- Stage 1: Python version check ---
echo
echo "[1/5] Checking Python version..."
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: '$PY' not found on PATH. Install Python 3.11+ first," >&2
    echo "       or set PYTHON=/path/to/python3.11 and re-run." >&2
    exit 1
fi

PY_VER=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_OK=$("$PY" -c 'import sys; print(1 if sys.version_info >= (3, 11) else 0)')
if [[ "$PY_OK" != "1" ]]; then
    echo "ERROR: Python $PY_VER detected. CLARYON requires Python >= 3.11." >&2
    echo "       Set PYTHON=/path/to/python3.11 and re-run." >&2
    exit 1
fi
echo "      Python $PY_VER OK."

# --- Stage 2: Virtual env ---
echo
echo "[2/5] Setting up virtual environment (.venv)..."
if [[ -d .venv ]]; then
    echo "      .venv already exists - reusing."
else
    "$PY" -m venv .venv
    echo "      Created .venv."
fi
VENV_PY=".venv/bin/python"
VENV_PIP=".venv/bin/pip"

if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: $VENV_PY not found after venv creation." >&2
    exit 1
fi

# --- Stage 3: Build prereqs ---
echo
echo "[3/5] Upgrading pip / setuptools / wheel..."
"$VENV_PIP" install --upgrade pip setuptools wheel

# --- Stage 4: CLARYON + standard extras ---
echo
if [[ "$CPU_ONLY" == 1 ]]; then
    echo "[4/5] Installing CPU-only PyTorch wheels first (--cpu-only flag)..."
    "$VENV_PIP" install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
    echo "      Installing CLARYON in editable mode with [all] extras..."
    echo "      (PennyLane, XGBoost, LightGBM, CatBoost, TabPFN, SHAP, LIME, ...)"
    "$VENV_PIP" install -e ".[all]"
else
    echo "[4/5] Installing CLARYON in editable mode with [all] extras..."
    echo "      (PennyLane, PyTorch, XGBoost, LightGBM, CatBoost, TabPFN, SHAP, LIME, ...)"
    "$VENV_PIP" install -e ".[all]"
fi

# --- Stage 5: pyradiomics (optional, with build-isolation workaround) ---
echo
if [[ "$INSTALL_RADIOMICS" == 1 ]]; then
    echo "[5/5] Installing pyradiomics (with --no-build-isolation workaround)..."
    echo "      pyradiomics 3.x's setup.py imports numpy at the top level, so the"
    echo "      isolated PEP 517 build env fails. We pre-install numpy + versioneer,"
    echo "      then install pyradiomics with build isolation disabled."
    "$VENV_PIP" install numpy versioneer
    "$VENV_PIP" install -e ".[radiomics]" --no-build-isolation
else
    echo "[5/5] Skipping pyradiomics install (--no-radiomics)."
fi

# --- Smoke test ---
echo
echo "=========================================="
echo " Smoke test: importing claryon..."
echo "=========================================="
"$VENV_PY" -c "import claryon; print(f'CLARYON {claryon.__version__} imported OK')"

echo
echo "=========================================="
echo " Install complete."
echo
echo " Activate the env:    source .venv/bin/activate"
echo " Verify registry:     claryon list-models"
echo " Run an example:      claryon run -c configs/example_tabular.yaml"
echo "=========================================="
