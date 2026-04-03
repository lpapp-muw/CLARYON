#!/bin/bash
# Run QCNN models on 3 datasets (fastest first)
# Estimated: HCC ~2.5h, PSMA-11 ~2.5h, Wisconsin ~15h
set -e
cd ~/claryon
source .venv/bin/activate

echo "=========================================="
echo "QCNN Benchmark - Started: $(date)"
echo "=========================================="

echo ""
echo "=== HCC (165 samples, ~2.5h) ==="
claryon -v run -c configs/eanm_abstract/qcnn_hcc_q8.yaml
echo "HCC done: $(date)"
cat Results/eanm_abstract/qcnn_hcc_q8/metrics_summary.csv
echo ""

echo "=== PSMA-11 (133 samples, ~2.5h) ==="
claryon -v run -c configs/eanm_abstract/qcnn_psma11_q8.yaml
echo "PSMA-11 done: $(date)"
cat Results/eanm_abstract/qcnn_psma11_q8/metrics_summary.csv
echo ""

echo "=== Wisconsin (569 samples, ~15h) ==="
claryon -v run -c configs/eanm_abstract/qcnn_wisconsin_q8.yaml
echo "Wisconsin done: $(date)"
cat Results/eanm_abstract/qcnn_wisconsin_q8/metrics_summary.csv
echo ""

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
echo ""
echo "=== SUMMARY ==="
for d in hcc psma11 wisconsin; do
  f="Results/eanm_abstract/qcnn_${d}_q8/metrics_summary.csv"
  if [ -f "$f" ]; then
    echo "--- $d ---"
    cat "$f"
    echo ""
  fi
done
