#!/bin/bash
cd ~/claryon
source .venv/bin/activate

for cfg in configs/eanm_abstract/wisconsin_q8.yaml \
           configs/eanm_abstract/hcc_q8.yaml \
           configs/eanm_abstract/psma11_q8.yaml; do
    echo "=========================================="
    echo "RUNNING: $cfg"
    echo "Started: $(date)"
    echo "=========================================="
    python -m claryon -v run -c "$cfg"
    echo "Finished: $(date)"
    echo ""
done

echo "ALL DONE: $(date)"
