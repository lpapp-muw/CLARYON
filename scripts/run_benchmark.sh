#!/bin/bash
# scripts/run_benchmark.sh — Run CLARYON benchmarks on included datasets
#
# Usage:
#   bash scripts/run_benchmark.sh                  # all 3 datasets
#   bash scripts/run_benchmark.sh wisconsin         # single dataset
#   bash scripts/run_benchmark.sh wisconsin hcc     # multiple datasets
#
# Available datasets: wisconsin, hcc, psma11
#
# Configs used: configs/eanm_abstract/<dataset>_q8.yaml
# These use max_features=8 (3 qubits) which is the recommended setting
# for quantum models on a CPU simulator.
#
# Results are written to Results/eanm_abstract/<dataset>_q8/
#
# Estimated runtimes (complexity: medium, 5-fold CV, 3 seeds, single CPU):
#   wisconsin:  2-4 hours
#   hcc:        2-4 hours
#   psma11:     2-4 hours
#   all three:  6-12 hours
#
# Tip: Run in a screen session for long benchmarks:
#   screen -S benchmark bash scripts/run_benchmark.sh
#   # Detach: Ctrl+A then D
#   # Reattach: screen -r benchmark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

ALL_DATASETS=("wisconsin" "hcc" "psma11")

# Parse arguments
if [ $# -eq 0 ]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("$@")
fi

# Map dataset names to config files
declare -A CONFIG_MAP
CONFIG_MAP[wisconsin]="configs/eanm_abstract/wisconsin_q8.yaml"
CONFIG_MAP[hcc]="configs/eanm_abstract/hcc_q8.yaml"
CONFIG_MAP[psma11]="configs/eanm_abstract/psma11_q8.yaml"

# Validate
for ds in "${DATASETS[@]}"; do
    cfg="${CONFIG_MAP[$ds]}"
    if [ -z "$cfg" ]; then
        echo "ERROR: Unknown dataset '$ds'. Available: ${ALL_DATASETS[*]}"
        exit 1
    fi
    if [ ! -f "$cfg" ]; then
        echo "ERROR: Config not found: $cfg"
        exit 1
    fi
done

echo "=========================================="
echo "CLARYON Benchmark"
echo "Datasets: ${DATASETS[*]}"
echo "Started:  $(date)"
echo "=========================================="
echo ""

for ds in "${DATASETS[@]}"; do
    cfg="${CONFIG_MAP[$ds]}"
    echo "=========================================="
    echo "DATASET: $ds"
    echo "CONFIG:  $cfg"
    echo "Started: $(date)"
    echo "=========================================="
    python -m claryon -v run -c "$cfg"
    echo "Finished $ds: $(date)"
    echo ""
done

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
echo ""
echo "Results:"
for ds in "${DATASETS[@]}"; do
    results_dir="Results/eanm_abstract/${ds}_q8"
    if [ -f "$results_dir/metrics_summary.csv" ]; then
        echo ""
        echo "--- $ds ---"
        cat "$results_dir/metrics_summary.csv"
    fi
done
