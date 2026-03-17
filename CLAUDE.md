# CLAUDE.md — Critical Fix: Quantum Preprocessing + Cleanup

## Project

**CLARYON** — CLassical-quantum AI for Reproducible Explainable OpeN-source medicine.

## READ FIRST

1. **WORKLOG.md** — Current state, hard facts (especially HF-031)
2. **claryon/pipeline.py** — Pipeline orchestrator (primary edit target)
3. **claryon/preprocessing/state.py** — PreprocessingState dataclass
4. **claryon/progress.py** — Progress display (exists but not wired)

## ============================================================
## CRITICAL BACKGROUND — READ BEFORE ANY CHANGES
## ============================================================
##
## HF-031: Z-score normalization before amplitude encoding DESTROYS
## quantum model performance. Experimentally verified:
##
##   With z-score:    kernel_svm 0.680, qcnn_muw 0.620
##   Without z-score: kernel_svm 1.000, qcnn_muw 0.970
##   XGBoost:         0.995 in both cases (scale-invariant)
##
## Root cause: Z-score shifts features to zero-mean with negative values.
## Amplitude encoding then L2-normalizes, completely distorting the
## geometric relationships the quantum kernel depends on. Classical
## tree-based models are unaffected because they are scale-invariant.
##
## Fix: Per-model-type preprocessing. Classical models get z-score + mRMR.
## Quantum models get mRMR only (amplitude encoding handles normalization).
##
## ============================================================

## ============================================================
## TASKS (execute in order, test after each)
## ============================================================

### Task 1: CRITICAL — Per-model-type preprocessing (skip z-score for quantum)

**File**: `claryon/pipeline.py` — inside the per-fold training loop

**Current behavior**: Z-score is applied to ALL models uniformly before training.

**Required behavior**: 
- `tabular` models: z-score + mRMR (unchanged)
- `tabular_quantum` models: mRMR ONLY, NO z-score
- `imaging` models: image normalization only (unchanged)

**Implementation**: Find the preprocessing block inside the fold loop. It currently does:
```python
# Something like:
X_train_z = apply_zscore(X_train, z_mean, z_std)
X_test_z = apply_zscore(X_test, z_mean, z_std)
X_train_sel = X_train_z[:, selected_features]
X_test_sel = X_test_z[:, selected_features]
```

Change to:
```python
# mRMR applies to ALL tabular models
X_train_sel = X_train[:, selected_features]
X_test_sel = X_test[:, selected_features]

if model_entry.type == "tabular_quantum":
    # Quantum: mRMR only, NO z-score
    # Amplitude encoding handles normalization
    X_train_use = X_train_sel
    X_test_use = X_test_sel
    logger.info("  Quantum model: skipping z-score (amplitude encoding normalizes)")
else:
    # Classical: z-score + mRMR
    X_train_use = apply_zscore(X_train_sel, z_mean_sel, z_std_sel)
    X_test_use = apply_zscore(X_test_sel, z_mean_sel, z_std_sel)
```

**IMPORTANT**: The z-score coefficients (z_mean, z_std) must still be COMPUTED and STORED in PreprocessingState for classical model inference. They just aren't applied to quantum models.

**PreprocessingState update**: Add a field or note indicating which preprocessing was applied:
```python
"preprocessing_applied": "mrmr_only"  # for quantum models
"preprocessing_applied": "zscore_mrmr" # for classical models
```

Or better: store both the raw selected features and the z-scored selected features, so inference knows what to apply based on model type.

**Test**:
```bash
# Run with preprocessing enabled — quantum should now perform well
python -m claryon -v run -c configs/iris_full_preprocess.yaml
# Verify: kernel_svm BACC ~1.0, qcnn_muw BACC ~0.95+, xgboost BACC ~0.99
cat Results/iris_preprocess/metrics_summary.csv
```

Also verify that PreprocessingState.json correctly records the mode.

### Task 2: Wire progress.py into pipeline.py

**File**: `claryon/pipeline.py` + `claryon/progress.py`

The `ProgressDisplay` class exists in `progress.py` with methods:
- `stage_start(name)` — prints `[N/8] Stage name...`
- `stage_end(summary)` — prints `✓ summary`  
- `model_progress(model_name, fold, total_folds)` — prints fold progress
- `print_summary_table(metrics_df)` — prints formatted results table

**Wire it**: At the top of `run_pipeline()`:
```python
from .progress import ProgressDisplay
progress = ProgressDisplay(verbosity=verbosity, n_stages=8)
```

Before each stage:
```python
progress.stage_start("Loading data")
# ... stage code ...
progress.stage_end(f"{n_samples} samples × {n_features} features")
```

Inside training loop:
```python
progress.model_progress(model_entry.name, fold_idx + 1, n_folds)
```

After evaluate stage:
```python
progress.print_summary_table(metrics_df)
```

**Verbosity**: Read it from the CLI args. `-v` = 1 (stages + summary), `-vv` = 2 (stages + per-fold + summary), no flag = 0 (summary only).

**Test**: Run any experiment with `-v` and verify formatted output appears.

### Task 3: Model/data type validation

**File**: `claryon/pipeline.py` — at the start of stage_train, before the model loop

**Add validation**:
```python
for model_entry in config.models:
    if not model_entry.enabled:
        continue
    
    # Block CNN on tabular-only data
    if model_entry.type == "imaging" and state.dataset.imaging_data is None:
        logger.error(
            "SKIPPING %s: model type 'imaging' requires imaging data "
            "(config.data.imaging). Cannot run CNN on tabular features.",
            model_entry.name
        )
        continue
    
    # Block tabular/quantum models on imaging-only data
    if model_entry.type in ("tabular", "tabular_quantum") and state.dataset.X is None:
        logger.error(
            "SKIPPING %s: model type '%s' requires tabular data "
            "(config.data.tabular). Cannot run on imaging data alone.",
            model_entry.name, model_entry.type
        )
        continue
    
    # Proceed with training...
```

**Test**: Create a config with `cnn_3d` but only tabular data. Verify it logs SKIPPING, doesn't crash, and other models still run.

### Task 4: Update license in pyproject.toml

**File**: `pyproject.toml`

Change:
```
license = {text = "BSD-3-Clause"}
```
To:
```
license = {text = "GPL-3.0-or-later"}
```

Also update any classifiers:
```python
"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
```

**Note**: The actual LICENSE file will be added via GitHub's license picker. This task only updates pyproject.toml metadata.

### Task 5: Trim benchmark downloader to 12 medical datasets

**File**: `claryon/benchmark/download_benchmark_datasets.py` (or wherever the downloader script lives — check `scripts/` too)

**Remove these 16 datasets** (non-medical):
- australian, credit-g, kc1, phoneme (Tier 1 non-medical)
- vehicle, segment, waveform-5000, steel-plates-fault, electricity, bank-marketing, adult (Tier 2 non-medical)
- wine-quality, dry-bean, rice-cammeo-osmancik, mushroom (Tier 4 non-medical)
- drug-classification (Tier 4 borderline — remove)

**Keep these 12 datasets**:
```python
OPENML_DATASETS = [
    {"name": "blood-transfusion",       "openml_id": 1464, "target": "Class",    "tier": 1},
    {"name": "diabetes",                "openml_id": 37,   "target": "class",    "tier": 1},
    {"name": "iris",                    "openml_id": 61,   "target": "class",    "tier": 0},  # demo only
]

UCI_DATASETS = [
    {"name": "wisconsin-breast-cancer", "uci_id": 17,   "tier": 3},
    {"name": "heart-failure",           "uci_id": 519,  "tier": 3},
    {"name": "cervical-cancer",         "uci_id": 383,  "tier": 3},
    {"name": "chronic-kidney-disease",  "uci_id": 336,  "tier": 3},
    {"name": "spect-heart",             "uci_id": 95,   "tier": 3},
    {"name": "mammographic-mass",       "uci_id": 161,  "tier": 3},
]

KAGGLE_DATASETS = [
    {"name": "hcc-survival",            "slug": "mrsantos/hcc-dataset",                    "tier": 3},
    {"name": "stroke-prediction",       "slug": "fedesoriano/stroke-prediction-dataset",   "tier": 3},
    {"name": "fetal-health",            "slug": "andrewmvd/fetal-health-classification",   "tier": 3},
]
```

Update the script header, docstring, and total count from 28 to 12.

### Task 6: Remove source_archive completely

```bash
rm -rf source_archive/
```

Add to `.gitignore`:
```
source_archive/
```

Also add `.claude/` to `.gitignore` if not already present.

### Task 7: Add HF-031 warning to structured methods.tex

**File**: `claryon/reporting/method_descriptions.yaml`

Add a preprocessing section entry:
```yaml
preprocessing:
  quantum_no_zscore:
    text: >
      For quantum models, z-score normalization was intentionally omitted
      prior to amplitude encoding. Amplitude encoding inherently
      L2-normalizes the feature vector, and prior z-score normalization
      distorts the geometric structure of the data in Hilbert space,
      degrading quantum kernel performance. Feature selection (mRMR)
      was applied to quantum models identically to classical models.
```

**File**: `claryon/reporting/structured_report.py`

In `_section_data()` or a new `_section_preprocessing()`, add this paragraph when quantum models are present in the config.

### Task 8: Add HF-031 warning to docs/model_guide.md

**File**: `docs/model_guide.md`

Add a prominent warning box:
```markdown
## Important: Preprocessing and Quantum Models

**Do NOT apply z-score normalization before quantum models.** 
CLARYON handles this automatically — quantum models receive 
mRMR-selected features without z-score, while classical models 
receive z-scored + mRMR-selected features. 

If you override preprocessing manually, be aware that z-score 
normalization before amplitude encoding destroys quantum kernel 
geometry and can reduce performance by 30-40%.
```

## ============================================================
## WORKFLOW
## ============================================================
##
## Execute tasks 1-8 in order. Test after each.
##
## After Task 1 (critical fix):
##   python -m claryon -v run -c configs/iris_full_preprocess.yaml
##   Verify: quantum BACC > 0.90, classical BACC > 0.99
##
## After Task 2 (progress):
##   python -m claryon -v run -c configs/iris_classical.yaml
##   Verify: formatted stage progress appears
##
## After Task 3 (validation):
##   Create a config with cnn_3d + tabular-only data
##   Verify: SKIPPING message, no crash
##
## After all tasks:
##   python -m pytest tests/ -x -q --timeout=300
##   git tag v0.12.0-quantum-fix
##
## ============================================================

## Code rules

Same as always:
- `from __future__ import annotations`
- Type hints, Google docstrings, logging not print
- Never crash — log error, skip model, continue
- Predictions through io/predictions.py, separator `;`

## Fix loop protocol

5 attempts max per bug. If stuck, BLOCKED file.
