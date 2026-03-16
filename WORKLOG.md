# CLARYON — Work Log

**Purpose**: Cross-session continuity document. Read this FIRST at the start of every new chat.

---

## 1. Current State

| Field | Value |
|---|---|
| **Phase** | 10 — preprocessing pipeline + infrastructure |
| **Last completed item** | Quantum methods port (v0.9.0-quantum) + notebooks |
| **Next item to build** | Stateful preprocessing pipeline (mRMR, z-score, image norm, binary grouping) |
| **Blockers** | None |
| **CURRENT_PHASE** | 10 |
| **CURRENT_STEP** | preprocessing_pipeline |

---

## 2. What's Done

- Phases 0-7: full framework built (50 min autonomous)
- Phase 8: pipeline wiring (stages 5-7: evaluate, explain, report)
- Phase 9: 5 new quantum methods ported from Moradi papers, GDQ score, NIfTI pattern fix, notebooks, README
- Structured methods.tex generator with method_descriptions.yaml text registry
- Verified: iris classical + quantum, NIfTI CNN, full pipeline with SHAP/LIME/LaTeX

## 3. Hard Facts

### HF-001 through HF-020: See previous sessions.

### HF-021: Preprocessing must happen INSIDE the fold loop
**Fact**: Z-score normalization and mRMR feature selection must be fitted on the TRAINING FOLD ONLY. Parameters are stored per fold/seed and applied to the test fold. Fitting on the full dataset before splitting causes data leakage and inflated metrics. This is a non-negotiable scientific validity requirement.

### HF-022: mRMR algorithm specification
**Fact**: mRMR as used in this project:
1. Compute Spearman rank correlation matrix among features (on training fold)
2. Cluster features by redundancy: features with |ρ| > spearman_threshold are redundant
3. Within each redundancy cluster, keep the feature with highest |Spearman ρ| to the target label
4. Default threshold: 0.8
5. Guard: skip if n_features <= 4
This is NOT the Peng et al. mutual-information-based mRMR. It is a correlation-based variant standard in radiomics research (Papp et al. 2018+).

### HF-023: PreprocessingState per fold
**Fact**: Each fold produces its own PreprocessingState containing:
- z-score μ, σ (per feature)
- Selected feature indices (from mRMR)
- Selected feature names
- Image normalization params (min, max) if applicable
Stored as `preprocessing_state.json` in the fold results directory. At inference time, load this state and apply before prediction.

### HF-024: Binary grouping is NOT one-vs-rest decomposition
**Fact**: Binary grouping is a user-defined relabeling of multiclass data into binary. User specifies which original labels map to positive (1) and which to negative (0). Example: ISUP grades [3,4] → positive, [1,2] → negative. This happens ONCE before splitting, not per-fold. The resulting dataset is binary throughout.

### HF-025: Image normalization modes
**Fact**: Two modes:
- per_image: each volume independently scaled to [0, 1]
- cohort_global: compute min/max from all training volumes, apply to all (train + test)
User-controlled via config. cohort_global params stored in PreprocessingState.

### HF-026: mRMR → encoding → n_qubits chain
**Fact**: The full chain is:
- Original features (e.g., 306 radiomics)
- After mRMR: reduced features (e.g., 24)
- After amplitude encoding: pad_len = next_pow2(24) = 32 → 5 qubits
- n_qubits auto-derived from pad_len (already implemented)
This chain must flow correctly through the pipeline. mRMR output is the input to encoding.

---

## 4. TODO for This Session

14 tasks — see CLAUDE.md for full specifications:

1. PreprocessingState dataclass (save/load/apply)
2. mRMR feature selection implementation
3. Z-score normalization (fit on train, apply to test)
4. Image normalization (per-image / cohort-global)
5. Binary grouping config + implementation
6. Preprocessing config in schema
7. **Rewire pipeline.py** (preprocessing inside fold loop)
8. BibTeX file generation
9. Hybrid method description fallback (model class attribute)
10. Remove DICOM from config schema
11. Ensemble reporting in methods + results tex
12. Update param_descriptions in method_descriptions.yaml
13. Example config exercising full pipeline
14. Update .gitignore

---

## 5. References

- Moradi S, ..., Papp L. Sci Rep 12, 1851 (2022). doi:10.1038/s41598-022-05971-9
- Moradi S, ..., Papp L. Eur J Nucl Med Mol Imaging 50, 3826-3837 (2023). doi:10.1007/s00259-023-06362-6
- Papp L, et al. "Quantum Convolutional Neural Networks for Predicting ISUP Grade risk in [68Ga]Ga-PSMA Primary Prostate Cancer Patients." Under revision.
