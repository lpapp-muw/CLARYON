# CLARYON — Work Log

**Purpose**: Cross-session continuity document. Read this FIRST.

---

## 1. Current State

| Field | Value |
|---|---|
| **Phase** | 11 — final polish |
| **Last completed item** | Preprocessing pipeline (v0.10.0-preprocess) |
| **Next item to build** | Model presets, auto mode, resource safety, GDQ framework, CLI, docs, cleanup |
| **Blockers** | None |
| **CURRENT_PHASE** | 11 |
| **CURRENT_STEP** | presets |

---

## 2. What's Done (all verified)

- Phases 0-7: full framework (50 min)
- Phase 8: pipeline wiring
- Phase 9: 5 quantum methods from Moradi papers, GDQ basic score, NIfTI pattern fix, notebooks
- Phase 10: mRMR feature selection, z-score normalization, binary grouping, image normalization, PreprocessingState per fold, BibTeX, hybrid method descriptions
- 183 tests passing

---

## 3. Hard Facts

HF-001 through HF-026: See previous sessions.

### HF-027: Quantum model defaults are smoke-test parameters
**Fact**: Current defaults (10-15 epochs, lr 0.02) are tuned for CI speed, not convergence. On real radiomics data (306 features, noisy signal), quantum models with these defaults will underperform classical models. Nuclear medicine users unfamiliar with quantum ML will draw false conclusions. Presets (quick/small/medium/large/exhaustive) + auto mode are essential for scientific validity.

### HF-028: Resource estimation formulas
**Fact**: Quantum circuit simulation cost is O(2^n_qubits) per forward pass. Kernel SVM builds N² matrix. QCNN gradient costs ~3× forward (parameter shift rule). SWAP test uses 2n+1 qubits. These formulas drive both auto mode and OOM protection.

### HF-029: Huang et al. 2021 GDQ framework
**Fact**: The full quantum advantage assessment requires three quantities: geometric difference g(K^C||K^Q), model complexity s_K(N) = y^T K^{-1} y, and effective dimension d = rank(K_Q). g alone is necessary but not sufficient. The full flowchart (Fig 1 of Huang et al.) determines: classical_sufficient / quantum_advantage_likely / inconclusive. Reference: Nature Communications 12, 2631 (2021).

### HF-030: Model persistence gap
**Fact**: Pipeline currently does not call model.save() after training. PreprocessingState is saved but models are not. Inference on new data requires both. Also missing: claryon infer subcommand.

---

## 4. TODO — 20 tasks in 5 groups

See CLAUDE.md for full specifications. Summary:

**Group A**: Presets + auto mode + safety (T1-T3)
**Group B**: Model save/load + inference + provenance (T4-T6)
**Group C**: CLI progress + banner + SHAP plots + GDQ framework (T7-T10)
**Group D**: Documentation (T11-T15)
**Group E**: Cleanup + README + final test (T16-T20)
