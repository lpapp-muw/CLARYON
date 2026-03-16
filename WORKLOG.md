# CLARYON — Work Log

**Purpose**: Cross-session continuity document. Read this FIRST at the start of every new chat.

---

## 1. Current State

| Field | Value |
|---|---|
| **Phase** | 8 — all phases built, pipeline wiring incomplete |
| **Last completed item** | Manual verification: classical + quantum + demo configs working |
| **Next item to build** | Wire pipeline stages 5-7 (evaluate, explain, report) + NIfTI data path in pipeline |
| **Blockers** | None |
| **Open questions** | None |
| **CURRENT_PHASE** | 8 |
| **CURRENT_STEP** | pipeline_wiring |

---

## 2. Session Log

### Sessions 1-2 — 2026-03-16

Project setup, naming (CLARYON), REQUIREMENTS.md v0.3.1, IMPLEMENTATION_PLAN.md v0.1.1.

### Session 3 — 2026-03-16 (Claude Code autonomous build)

**Completed**: All 8 phases built autonomously in ~50 minutes.

| Phase | Tag | Tests | Key Deliverables |
|---|---|---|---|
| 0 | v0.0.0 | 50 | pyproject.toml, registry, determinism, io/base, predictions, config_schema, CLI, pipeline skeleton |
| 1 | v0.1.0 | 102 | io/tabular, io/nifti, io/tiff, io/fdb_ldb, encoding/amplitude, preprocessing (tabular_prep, splits, radiomics, image_prep) |
| 2 | v0.2.0 | 119 | XGBoost, LightGBM, CatBoost, MLP, TabPFN, DEBI-NN wrapper, stubs, ensemble, pipeline stages 1-4 |
| 3 | v0.3.0 | 127 | Quantum kernel SVM, QCNN MUW, QCNN ALT, angle encoding, VQC/hybrid stubs |
| 4 | v0.4.0 | 132 | 2D/3D CNN (PyTorch), late fusion integration |
| 5 | v0.5.0 | 139 | SHAP, LIME, GradCAM stub, explainability utilities |
| 6 | v0.6.0 | 163 | 12 registered metrics, Friedman/Nemenyi, bootstrap CI, figures, LaTeX/Markdown reporting |
| 7 | v0.7.0 | 165 | GitHub Actions CI, Docker/GPU/Singularity, example configs, full integration tests |

248 tracked files, 165 tests passing, 19+ commits on linear main history.

### Session 4 — 2026-03-16 (Manual verification + fixes)

**Issues found and fixed**:

1. `claryon/__main__.py` was missing → created (enables `python -m claryon`)
2. `pipeline.py` only imported classical models → patched to import quantum + CNN modules
3. `pipeline.py` had no amplitude encoding for quantum models → patched: detects `model_entry.type == "tabular_quantum"`, calls `amplitude_encode_matrix()`, passes encoded data to fit/predict
4. Config `n_qubits` must match encoding `pad_len` → iris has 4 features, pad_len=4, requires n_qubits=2 (not 3)
5. CLI `-vv` flag must go before subcommand: `python -m claryon -v run -c ...`

**Verified working**:
- `python -m claryon -v run -c configs/iris_classical.yaml` — XGBoost, LightGBM, CatBoost × 5 folds, 10 seconds
- `python -m claryon -v run -c configs/iris_quantum.yaml` — kernel_svm (12s), qcnn_muw (46s), qcnn_alt (44s)
- Predictions output: `Key;Actual;Predicted;P0;P1;Fold;Seed` — correct semicolon format
- 165 built-in tests still passing

**Still stub/unwired in pipeline.py**:
- `stage_load_data()`: only handles tabular. NIfTI/TIFF imaging path not wired.
- `stage_preprocess()`: passthrough. No imputation/scaling/radiomics extraction.
- `stage_evaluate()`: stub. Metrics module exists (`evaluation/metrics.py`) but not called.
- `stage_explain()`: stub. SHAP/LIME modules exist but not called.
- `stage_report()`: stub. LaTeX/Markdown modules exist but not called.
- No automatic n_qubits derivation from encoding — user must manually match in config.

**Commits**:
- `70ad976` — fix: __main__.py, pipeline quantum/CNN imports, amplitude encoding, demo configs
- `3514267` — demo: iris classical + quantum + configs verified

---

## 3. Hard Facts & Lessons Learned

### HF-001 through HF-008
(See previous sessions — retained, not repeated here for brevity)

### HF-009: pyradiomics broken packaging
**Fact**: pyradiomics 3.1.0 has inconsistent metadata. 3.0.1 needs versioneer. Neither installs with build isolation.
**Fix**: `pip install numpy versioneer && pip install pyradiomics==3.0.1 --no-build-isolation`

### HF-010: Amplitude encoding ↔ n_qubits coupling
**Fact**: If data has N features, amplitude encoding pads to next power of 2. The model's `n_qubits` parameter must equal `log2(pad_len)`. Mismatch causes PennyLane error: "State must be of length 2^n_qubits; got length pad_len."
**Fix needed**: Pipeline should derive n_qubits from the encoding result automatically, overriding the config value. Currently the user must manually match them.

### HF-011: Pipeline only loads tabular data
**Fact**: `stage_load_data()` only handles `config.data.tabular`. The `config.data.imaging` path (NIfTI/TIFF) is defined in the schema but not implemented in the pipeline. The NIfTI loader module (`claryon/io/nifti.py`) works — it just isn't called from the pipeline.

### HF-012: Evaluate/Explain/Report stages are stubs
**Fact**: The individual modules exist and pass their own tests:
- `evaluation/metrics.py` — 12 registered metrics
- `evaluation/comparator.py` — Friedman/Nemenyi, bootstrap CI
- `evaluation/figures.py` — ROC, confusion matrix, CD diagram
- `explainability/shap_.py`, `explainability/lime_.py` — working
- `reporting/latex_report.py`, `reporting/markdown_report.py` — working
But `pipeline.py` stages 5/6/7 are one-liner stubs that log and return.

### HF-013: Server environment
**Fact**: Ubuntu server `omega`, user `laszlo`, Python 3.11, PyTorch CPU-only, PennyLane installed, pyradiomics installed (v3.0.1 via --no-build-isolation). Git branch: main.

---

## 4. File Delivery Tracker

| File | Version | Date | Status |
|---|---|---|---|
| All Phase 0-7 code | v0.7.0 | 2026-03-16 | Built by Claude Code, on server |
| `claryon/__main__.py` | v1 | 2026-03-16 | Manual fix, committed |
| `claryon/pipeline.py` | patched | 2026-03-16 | Quantum/CNN imports + amplitude encoding added |
| `configs/iris_classical.yaml` | v1 | 2026-03-16 | Verified working |
| `configs/iris_quantum.yaml` | v1 | 2026-03-16 | Verified working (n_qubits=2) |
| `configs/nifti_cnn.yaml` | v1 | 2026-03-16 | Not yet tested |

---

## 5. Next Session Checklist

Wire the remaining pipeline stages. The modules exist — they need to be called from pipeline.py with correct data flow. Specific tasks:

1. Wire `stage_load_data()` for imaging data (NIfTI/TIFF via config.data.imaging)
2. Wire `stage_preprocess()` — at minimum: tabular scaling, optional radiomics extraction
3. Wire `stage_evaluate()` — read Predictions.csv per model/fold, compute metrics, aggregate
4. Wire `stage_explain()` — run SHAP/LIME on configured models
5. Wire `stage_report()` — generate Markdown/LaTeX from aggregated results
6. Auto-derive n_qubits from amplitude encoding (eliminate manual config mismatch)
7. Test all wired stages end-to-end
