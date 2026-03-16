# CLAUDE.md — Pipeline Wiring Session

## Project

**CLARYON** — CLassical-quantum AI for Reproducible Explainable OpeN-source medicine.
All 8 build phases are DONE. This session wires the remaining pipeline stages.

## READ FIRST

1. **WORKLOG.md** — Current state, what's done, what's stub, hard facts
2. **claryon/pipeline.py** — The file you're primarily editing
3. **REQUIREMENTS.md** — Authoritative spec

## What's already working

- `python -m claryon -v run -c configs/iris_classical.yaml` — XGBoost/LightGBM/CatBoost × 5 folds
- `python -m claryon -v run -c configs/iris_quantum.yaml` — kernel_svm, qcnn_muw, qcnn_alt with amplitude encoding
- 165 unit tests passing
- All model modules registered and functional
- All evaluation/explainability/reporting modules exist and pass their own tests

## What needs to be wired

These modules EXIST and WORK individually. They need to be called from `pipeline.py`.

### Task 1: NIfTI/imaging data loading in stage_load_data()

Currently `stage_load_data()` only handles `config.data.tabular`. Wire the imaging path:

```python
# The loader exists:
from claryon.io.nifti import load_nifti_dataset
# Config schema already supports:
# config.data.imaging.path, config.data.imaging.format, config.data.imaging.mask_pattern
```

When imaging data is loaded, flatten the masked voxels into a feature vector per sample, producing a Dataset with X shape (n_samples, n_voxels). For early fusion with tabular data, concatenate features.

### Task 2: Auto-derive n_qubits from encoding (HF-010)

Currently the pipeline amplitude-encodes data and passes it to quantum models, but the model's `n_qubits` comes from the YAML config. If the user sets `n_qubits: 3` but the encoding produces pad_len=4 (2 qubits), PennyLane crashes.

Fix: after amplitude encoding, override the model's n_qubits with `enc_info.n_qubits`. The encoding result is the authoritative source for qubit count.

Find this block in pipeline.py:
```python
if model_entry.type == "tabular_quantum":
    from .encoding.amplitude import amplitude_encode_matrix
    X_tr_use, enc_info = amplitude_encode_matrix(X_train)
    X_te_use, _ = amplitude_encode_matrix(X_test, pad_len=enc_info.encoded_dim)
```

Add after it:
```python
    # Override n_qubits from encoding (authoritative)
    model_entry.params["n_qubits"] = enc_info.n_qubits
```

But note: ModelEntry is a Pydantic model with a frozen `params` dict. You may need to construct params before passing to model_cls().

### Task 3: Wire stage_evaluate()

The metrics module is at `claryon/evaluation/metrics.py`. It has registered metrics.

Wire it to:
1. After stage_train completes, iterate over `state.results` (model → list of fold results)
2. For each model/seed/fold, read the Predictions.csv
3. Compute all metrics from `config.evaluation.metrics`
4. Aggregate across folds (mean ± std)
5. Store in `state.results` or write a summary CSV

Key functions to use:
```python
from claryon.io.predictions import read_predictions
from claryon.evaluation.metrics import get as get_metric  # or use registry
```

### Task 4: Wire stage_explain()

The explainability modules are at:
- `claryon/explainability/shap_.py`
- `claryon/explainability/lime_.py`

Wire based on `config.explainability.shap` and `config.explainability.lime` booleans.

Explainability runs AFTER training — it needs a fitted model and test data. Use the model objects from stage_train (you may need to persist them in PipelineState).

### Task 5: Wire stage_report()

The reporting modules are at:
- `claryon/reporting/markdown_report.py`
- `claryon/reporting/latex_report.py`

Wire based on `config.reporting.markdown` and `config.reporting.latex` booleans.

Reports consume the aggregated metrics from stage_evaluate. They write to `state.results_dir`.

### Task 6: Wire stage_preprocess()

At minimum, wire optional tabular preprocessing:
- `claryon/preprocessing/tabular_prep.py` — scaling, imputation
- `claryon/preprocessing/radiomics.py` — pyradiomics extraction (if config.data.radiomics.extract is True)

## ============================================================
## WORKFLOW
## ============================================================
##
## For each task (1-6):
##   1. Read the existing module code to understand its API
##   2. Edit pipeline.py to call it with correct arguments
##   3. Write/update integration tests
##   4. Run: python -m pytest tests/ -x -q --timeout=300
##   5. Run an end-to-end experiment to verify
##   6. Commit: git add -A && git commit -m "wire: <description>"
##
## After all tasks:
##   1. Run full test suite: python -m pytest tests/ -q --timeout=600
##   2. Run all 3 demo experiments:
##      python -m claryon -v run -c configs/iris_classical.yaml
##      python -m claryon -v run -c configs/iris_quantum.yaml
##      python -m claryon -v run -c configs/nifti_cnn.yaml
##   3. Verify Results/ contains predictions + metrics + reports
##   4. git tag v0.8.0-wired
##   5. Update WORKLOG.md
##
## ============================================================

## Code rules (same as build phase)

- `from __future__ import annotations` at top of every .py file
- Type hints on all function signatures
- Google-style docstrings on all public functions
- `import logging; logger = logging.getLogger(__name__)` — never `print()`
- Deterministic: every stochastic operation seeded
- All predictions through `claryon/io/predictions.py`
- CSV separator: `;`

## Fix loop protocol

If tests fail:
1. Read full error. Log: `echo "FIX <n>: <diagnosis>" >> fix_log.txt`
2. Apply minimal fix. Re-run.
3. If 5 attempts fail: commit as BLOCKED, touch BLOCKED file, stop.

## Stop conditions

- BLOCKED file (5 fix attempts exhausted)
- Architectural ambiguity REQUIREMENTS.md doesn't resolve
- Need to modify a model module's API (ask first)

## Key file locations

```
claryon/pipeline.py              ← PRIMARY EDIT TARGET
claryon/io/nifti.py              ← NIfTI loader (Task 1)
claryon/io/predictions.py        ← Read/write predictions
claryon/encoding/amplitude.py    ← amplitude_encode_matrix() (Task 2)
claryon/evaluation/metrics.py    ← Metric functions (Task 3)
claryon/evaluation/comparator.py ← Statistical tests (Task 3)
claryon/evaluation/figures.py    ← Plot generators (Task 3)
claryon/explainability/shap_.py  ← SHAP wrapper (Task 4)
claryon/explainability/lime_.py  ← LIME wrapper (Task 4)
claryon/reporting/markdown_report.py ← Markdown generator (Task 5)
claryon/reporting/latex_report.py    ← LaTeX generator (Task 5)
claryon/preprocessing/tabular_prep.py ← Scaling/imputation (Task 6)
claryon/preprocessing/radiomics.py    ← PyRadiomics wrapper (Task 6)
claryon/config_schema.py         ← Config structure (read-only reference)
tests/test_integration/          ← Integration tests
configs/iris_classical.yaml      ← Test config (classical)
configs/iris_quantum.yaml        ← Test config (quantum)
configs/nifti_cnn.yaml           ← Test config (CNN)
```

## Demo experiments for verification

```bash
# Classical (should show metrics in output after wiring)
python -m claryon -v run -c configs/iris_classical.yaml

# Quantum (should show metrics + optional SHAP)
python -m claryon -v run -c configs/iris_quantum.yaml

# NIfTI CNN (should load NIfTI, train 3D CNN, show metrics)
python -m claryon -v run -c configs/nifti_cnn.yaml
```
