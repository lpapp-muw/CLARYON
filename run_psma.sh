#!/bin/bash
cd ~/claryon
source .venv/bin/activate
echo "Started PSMA: $(date)"
python -m claryon -v run -c configs/eanm_abstract/psma11.yaml
echo "Finished PSMA: $(date)"
