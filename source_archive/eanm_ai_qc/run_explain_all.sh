#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Ensure package imports work when running example scripts.
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "[EXPLAIN] Running SHAP+LIME on all stored models under Results/"
echo "[EXPLAIN] This can take a long time (quantum simulator + many model evaluations)."

python3 examples/run_explain_all.py --results-dir Results
