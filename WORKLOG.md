# CLARYON — Work Log

**Purpose**: Cross-session continuity document. Read this FIRST.

---

## 1. Current State

| Field | Value |
|---|---|
| **Phase** | 12 — quantum preprocessing fix + cleanup |
| **Last completed item** | Bugfix session v0.11.1, cleanup v0.11.2 |
| **Next item to build** | Critical: per-model-type preprocessing (skip z-score for quantum) |
| **Blockers** | None |
| **CURRENT_PHASE** | 12 |
| **CURRENT_STEP** | quantum_zscore_fix |

---

## 2. What's Done

- Phases 0-7: framework (50 min)
- Phase 8: pipeline wiring
- Phase 9: 5 quantum methods from Moradi papers, GDQ, NIfTI fix, notebooks
- Phase 10: mRMR, z-score, binary grouping, PreprocessingState per fold, BibTeX
- Phase 11: presets, auto mode, OOM safety, model save/load, inference, CLI banner, SHAP plots, GDQ framework, 5 docs, cleanup
- Phase 11.1: 8 bugfixes (multiclass AUC, structured methods.tex, float precision, ± std)
- Phase 11.2: cleanup (session files → docs/development, singularity version, references.bib corrected)
- 183 tests passing, v0.11.2-clean

---

## 3. Hard Facts

### ALL PREVIOUS HF-001 through HF-030 apply.

### HF-031: Z-score before amplitude encoding DESTROYS quantum performance [CRITICAL]

**Experimentally verified on iris binary (150 samples, 4 features, 5-fold CV)**:

| Config | Preprocessing | kernel_svm BACC | qcnn_muw BACC | xgboost BACC |
|---|---|---|---|---|
| iris_full (no preprocess) | None | 1.000 | 1.000 | 0.995 |
| iris_preprocess (z-score ON) | z-score + mRMR | 0.680 | 0.620 | 0.995 |
| iris_medium (z-score ON) | z-score + mRMR | 0.680 | 0.620 | 0.995 |
| iris_medium_nopreprocess | None | 1.000 | 0.970 | 0.995 |

**Root cause**: Z-score shifts features to zero-mean with negative values. Amplitude encoding then L2-normalizes, distorting the geometric relationships the quantum kernel depends on. Classical tree-based models are scale-invariant and unaffected.

**Fix**: Pipeline must apply z-score ONLY to classical models. Quantum models receive mRMR-selected features without z-score. Amplitude encoding handles normalization for quantum.

**This is a scientifically important finding.** It means: any paper applying standard preprocessing before quantum amplitude encoding may report artificially degraded quantum performance. Must be documented in methods.tex, model_guide.md, and any publications.

---

## 4. TODO — 8 tasks

| # | Task | Priority | Estimated time |
|---|---|---|---|
| 1 | Per-model-type preprocessing (skip z-score for quantum) | **CRITICAL** | 10 min |
| 2 | Wire progress.py into pipeline.py | High | 10 min |
| 3 | Model/data type validation (block CNN on tabular, etc.) | High | 5 min |
| 4 | Update pyproject.toml license to GPL-3.0-or-later | High | 1 min |
| 5 | Trim benchmark downloader from 28 to 12 medical datasets | Medium | 5 min |
| 6 | Remove source_archive/, add to .gitignore | Medium | 1 min |
| 7 | Add HF-031 warning to method_descriptions.yaml + structured_report.py | Medium | 5 min |
| 8 | Add HF-031 warning to docs/model_guide.md | Medium | 2 min |

**Estimated total**: ~40 minutes
