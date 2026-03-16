#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p Results

echo "[EXAMPLE] Repo root: ${ROOT_DIR}"
echo "[EXAMPLE] Project: EANM-AI-QC"
echo "[EXAMPLE] Building real tabular CSVs from demo_data/tabular/raw/{FDB,LDB}.csv"
python3 examples/build_tabular_from_fdb_ldb.py

echo
echo "[EXAMPLE] Generating synthetic NIfTI mockups"
python3 examples/make_synthetic_nifti.py

echo
echo "[EXAMPLE] RUN (tabular): methods = pl_kernel_svm + pl_qcnn_alt + pl_qcnn_muw (NO SHAP/LIME)"
python3 qnm_qai.py run \
  --input demo_data/tabular/real_train.csv \
  --infer demo_data/tabular/real_infer.csv \
  --input-type tabular \
  --methods pl_kernel_svm,pl_qcnn_alt,pl_qcnn_muw \
  --results-dir Results \
  --seed 0 \
  --test-size 0.25 \
  --qcnn-epochs 15 \
  --qcnn-lr 0.02 \
  --qcnn-batch-size 16 \
  --qcnn-init-scale 0.1 \
  --max-samples-per-method 80 \
  --no-explain

echo
echo "[EXAMPLE] RUN (nifti masked): methods = pl_qcnn_alt + pl_qcnn_muw (NO SHAP/LIME)"
python3 qnm_qai.py run \
  --input demo_data/nifti_masked \
  --input-type nifti \
  --methods pl_qcnn_alt,pl_qcnn_muw \
  --results-dir Results \
  --seed 0 \
  --test-size 0.30 \
  --qcnn-epochs 15 \
  --qcnn-lr 0.02 \
  --qcnn-batch-size 16 \
  --qcnn-init-scale 0.1 \
  --max-samples-per-method 40 \
  --no-explain

echo
echo "[EXAMPLE] Result files:"
find Results -maxdepth 2 -type f -name "*__results.csv" -print

echo "Done"
