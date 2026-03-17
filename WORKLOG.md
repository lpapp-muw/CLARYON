# CLARYON — Work Log

**Purpose**: Cross-session continuity document. Read this FIRST.

---

## 1. Current State

| Field | Value |
|---|---|
| **Phase** | 12 — quantum preprocessing fix + cleanup — COMPLETE |
| **Last completed item** | All 8 tasks done, v0.12.0-quantum-fix |
| **Next item to build** | None — ready for tagging |
| **Blockers** | None |
| **CURRENT_PHASE** | 12 |
| **CURRENT_STEP** | done |

---

## 2. What's Done

- Phases 0-7: framework (50 min)
- Phase 8: pipeline wiring
- Phase 9: 5 quantum methods from Moradi papers, GDQ, NIfTI fix, notebooks
- Phase 10: mRMR, z-score, binary grouping, PreprocessingState per fold, BibTeX
- Phase 11: presets, auto mode, OOM safety, model save/load, inference, CLI banner, SHAP plots, GDQ framework, 5 docs, cleanup
- Phase 11.1: 8 bugfixes (multiclass AUC, structured methods.tex, float precision, ± std)
- Phase 11.2: cleanup (session files → docs/development, singularity version, references.bib corrected)
- Phase 12: quantum preprocessing fix + 7 cleanup tasks (see below)
- 298 tests passing, v0.12.0-quantum-fix

### Phase 12 Tasks (all completed)

1. **Per-model-type preprocessing** — quantum models skip z-score, get mRMR only. Classical models get zscore+mRMR. PreprocessingState records `preprocessing_applied` field. Verified: kernel_svm BACC 1.000, qcnn_muw BACC 1.000, xgboost BACC 0.995.
2. **Progress display wired** — ProgressDisplay from progress.py integrated into run_pipeline(). Stages show [N/8] labels, fold-level progress at -vv, summary table at end.
3. **Model/data type validation** — imaging models blocked on tabular-only data, tabular/quantum models blocked on imaging-only data. Logged and skipped gracefully.
4. **License updated** — pyproject.toml changed from BSD-3-Clause to GPL-3.0-or-later, added GPLv3+ classifier.
5. **Benchmark downloader created** — claryon/benchmark/download_benchmark_datasets.py with 12 medical datasets (3 OpenML, 6 UCI, 3 Kaggle).
6. **source_archive cleanup** — already removed; .claude/ added to .gitignore.
7. **HF-031 in methods** — quantum_no_zscore text added to method_descriptions.yaml, auto-included in structured_report.py when quantum models present.
8. **HF-031 in model_guide** — warning box added to docs/model_guide.md before Common Pitfalls section.

---

## 3. Hard Facts

### ALL PREVIOUS HF-001 through HF-030 apply.

### HF-031: Z-score before amplitude encoding DESTROYS quantum performance [FIXED]

**Experimentally verified on iris binary (150 samples, 4 features, 5-fold CV)**:

| Config | Preprocessing | kernel_svm BACC | qcnn_muw BACC | xgboost BACC |
|---|---|---|---|---|
| iris_full (no preprocess) | None | 1.000 | 1.000 | 0.995 |
| iris_preprocess (z-score ON, old) | z-score + mRMR | 0.680 | 0.620 | 0.995 |
| iris_preprocess (FIXED) | per-model-type | 1.000 | 1.000 | 0.995 |

**Root cause**: Z-score shifts features to zero-mean with negative values. Amplitude encoding then L2-normalizes, distorting the geometric relationships the quantum kernel depends on. Classical tree-based models are scale-invariant and unaffected.

**Fix implemented**: Pipeline applies z-score ONLY to classical models. Quantum models receive mRMR-selected features without z-score. PreprocessingState.preprocessing_applied records which mode was used.

---

## 4. TODO

None — all 8 tasks complete. Ready for `git tag v0.12.0-quantum-fix`.
