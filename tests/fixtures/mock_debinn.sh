#!/usr/bin/env bash
# Mock DEBI-NN binary for testing.
# Mimics the expected output format: writes Predictions.csv files.
# Usage: mock_debinn.sh <project_dir>

PROJECT_DIR="${1%/}"
if [ -z "$PROJECT_DIR" ]; then
    echo "ERROR: No project directory specified" >&2
    exit 1
fi

# Find dataset name from executionSettings.csv if it exists
DATASET="test_dataset"

# Create output structure
EXEC_DIR="$PROJECT_DIR/Executions-Finished/$DATASET-M0/Log/Fold-1"
mkdir -p "$EXEC_DIR"

# Write a minimal Predictions.csv
cat > "$EXEC_DIR/Predictions.csv" << 'PRED'
Key;Actual;Predicted;P0;P1
S0000;0;0;0.80000000;0.20000000
S0001;1;1;0.30000000;0.70000000
S0002;0;0;0.90000000;0.10000000
S0003;1;1;0.25000000;0.75000000
S0004;0;1;0.45000000;0.55000000
PRED

echo "Mock DEBI-NN completed successfully"
exit 0
