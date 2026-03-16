# CLARYON — Work Log

**Purpose**: Cross-session continuity document. Read this FIRST at the start of every new chat. Update after every major implementation step. Drop the updated version back into the project.

---

## How To Use This File

1. **New chat starts**: Claude reads this file via project knowledge → knows exactly where we stopped, what's done, what's next, and what pitfalls to avoid.
2. **During implementation**: After each major step (file created, test passing, phase gate cleared), append an entry below.
3. **End of session**: User downloads the updated WORKLOG.md and drops it into the project, replacing the previous version.
4. **Hard facts**: Any non-obvious discovery (dependency conflict, API quirk, performance finding, design decision made mid-implementation) goes into Section 3. These never get deleted — they accumulate.

---

## 1. Current State

| Field | Value |
|---|---|
| **Phase** | CURRENT_PHASE: 5 |
| **Last completed item** | Phase 4 gate passed — 132 tests, 2D/3D CNN + late fusion working |
| **Next item to build** | `claryon/explainability/shap_.py` (Phase 5, item 1) |
| **Blockers** | None |
| **Open questions** | None |
| **Active chat** | Chat 3 (2026-03-16) — Phase 0 complete, starting Phase 1 |

---

## 2. Session Log

### Session 1 — 2026-03-16

**Scope**: Project setup, codebase audit, requirements, implementation planning.

**Completed**:
- Catalogued all 16 Benchmark project files (config.py through competitor_log.txt)
- Catalogued all 25+ EANM-AI-QC project files (qnm_qai.py through notebooks)
- Identified and resolved 3 missing `__init__.py` files (io, models, explain sub-packages)
- Completeness audit: all imports resolve, no missing source files (only external: DEBI-NN C++ binary)
- REQUIREMENTS.md v0.3.0 — 18 sections + 2 appendices, 795 lines
- IMPLEMENTATION_PLAN.md v0.1.0 — 7 sections, full file disposition map, dependency chains, phase gates, build order
- WORKLOG.md v0.1.0 — this file

**Not started**: No implementation code written yet.

**Decisions made**:
- Prediction output: semicolon-separated, Benchmark-style (`Key;Actual;Predicted;P0;...;PK-1`)
- Fusion: early (flatten), late, and intermediate (planned) all supported
- Ensemble: softmax averaging for classification, raw mean for regression
- Dependency conflicts: documentation-only for MVP (separate venvs, shared CV splits on disk)
- Experiment tracking: file-based provenance for MVP, external systems (MLflow/W&B) deferred
- Testing: 4 levels (unit, smoke, integration, benchmark regression), CI via GitHub Actions
- n8n/Airflow: future, but CLI designed to be pipeline-friendly

**Issues encountered**: 
- Claude project cannot hold multiple files with the same name (`__init__.py` × 4). Workaround: attach duplicates as inline documents in chat, not as project files.

### Session 2 — 2026-03-16

**Scope**: Project naming decision.

**Completed**:
- Evaluated ~40 candidate acronyms across 4 rounds of generation
- Web-verified each candidate for namespace collisions (GitHub, PyPI, medical AI, quantum ML, imaging software)
- Killed all initial candidates (QUASAR, ORACLE, MERLIN, MANTIS, HELIX, NEXUS, QUIVER, QAIMS, PRISM, SPECTRA) due to significant collisions
- Second round candidates (RADIQ, COHERA, MIRAQ, etc.) evaluated; RADIQ rejected (radiology connotation alienates NM), COHERA rejected (sounds like pharma brand)
- Explored two-repo split (NUCTERA/OCTERA) — rejected as unnecessary maintenance burden for a single-maintainer project with modality-agnostic architecture
- Final decision: **CLARYON** — CLassical-quantum AI for Reproducible Explainable OpeN-source medicine
- "CLAREON" was first choice but collides with Alcon's $40B intraocular lens product line (Clareon PanOptix, Clareon Vivity). Y-variant dodges the trademark.
- Two-repo strategy locked: CLARYON = engineering codebase, EANM-AI-QC = educational hub
- Updated REQUIREMENTS.md §1.1 (project identity, two-repo strategy, package name)
- Updated REQUIREMENTS.md Appendix B D-6 (repo name decision marked DECIDED)
- Bulk-renamed all 91 `eanm_ai_qc`/`eanm-ai-qc` references in IMPLEMENTATION_PLAN.md to `claryon`
- Updated all three governing document headers

**Not started**: No implementation code written yet.

**Decisions made**:
- Package name: `claryon` (`pip install claryon`)
- EANM-AI-QC repo retained as educational hub (no code)
- CLARYON repo is the sole engineering artifact

### Session 3 — 2026-03-16

**Scope**: Phase 0 completion + Phase 1 build.

**Completed**:
- Verified all Phase 0 items already implemented from prior scaffold commit
- All 50 unit tests passing: registry, determinism, config_schema, io/base, io/predictions, encoding/base, models/base, explainability/base
- `pip install -e .` succeeds, `claryon --help` works
- Phase 0 validation script passes (TIER 1)
- Tagged v0.0.0
- Phase 0 gate: PASSED
- Implemented all Phase 1 modules: io/tabular.py, io/nifti.py, io/tiff.py, io/fdb_ldb.py, encoding/amplitude.py, preprocessing/tabular_prep.py, preprocessing/splits.py, preprocessing/radiomics.py, preprocessing/image_prep.py
- Created __init__.py for all sub-packages
- 52 new tests written (102 total passing)
- Phase 1 validation (Tier 1+2) passed
- Phase 1 gate: PASSED
- Tagged v0.1.0
- Implemented all Phase 2 modules: XGBoost, LightGBM, CatBoost, MLP, TabPFN, DEBI-NN wrapper, stubs (TabM, RealMLP, ModernNCA), ensemble, pipeline stages 1-4
- Fixed .gitignore models/ pattern that was excluding claryon/models/
- Classical pipeline integration test: config → load → split → train → predict → write
- 17 new tests (119 total passing)
- Phase 2 validation (Tier 1+2+3) passed
- Phase 2 gate: PASSED
- Tagged v0.2.0
- Implemented Phase 3: angle encoding, kernel_svm, qcnn_muw, qcnn_alt (PORT from [E]), VQC/hybrid stubs
- Quantum smoke tests: 4 qubits, 2 epochs, 10-20 samples — all passing
- 8 new tests (127 total)
- Phase 3 validation (Tier 1-4) passed
- Phase 3 gate: PASSED
- Tagged v0.3.0

---

## 3. Hard Facts & Lessons Learned

Permanent knowledge base. Never delete entries — only append.

### HF-001: __init__.py Upload Limitation
**Date**: 2026-03-16
**Context**: EANM-AI-QC has 4 `__init__.py` files in different sub-packages.
**Fact**: Claude project file system is flat — cannot hold multiple files with the same filename. Last upload overwrites previous.
**Workaround**: Attach as inline document in chat message. Claude sees all copies in the message context.
**Impact**: When rebuilding the project, these files must be generated from known contents, not uploaded as project files.

### HF-002: EANM-AI-QC Sub-Package Init Contents
**Date**: 2026-03-16
**Fact**: Verified contents of all 4 `__init__.py` files:
- `eanm_ai_qc/__init__.py`: `__version__ = '0.8.0'`
- `eanm_ai_qc/io/__init__.py`: re-exports `load_tabular_csv`, `load_nifti_dataset`, `load_nifti_for_inference`
- `eanm_ai_qc/models/__init__.py`: re-exports `PLAmplitudeKernelSVM`, `PLQCNN_MUW`, `PLQCNN_Alt`
- `eanm_ai_qc/explain/__init__.py`: re-exports `run_shap`, `run_lime`

### HF-003: Benchmark CSV Separator Convention
**Date**: 2026-03-16
**Fact**: All DEBI-NN and Benchmark project CSVs use semicolon (`;`) separator. Float format: `%.8f`. Key format: `S0000..S{n-1}`.
**Decision**: The combined project adopts `;` as the universal CSV separator (REQ §8.4).

### HF-004: Quantum Model Probability Calibration
**Date**: 2026-03-16  
**Context**: EANM-AI-QC `runner.py` lines 239-243.
**Fact**: Quantum models (especially QCNN) produce probabilities clustered near 0.5. Fixed threshold at 0.5 gives poor balanced accuracy. The existing codebase solves this with Youden's J threshold optimization on train data.
**Decision**: Preserve this pattern in the combined codebase. `select_threshold_balanced_accuracy()` from metrics.py is critical for quantum model evaluation.

### HF-005: DEBI-NN Binary Interface
**Date**: 2026-03-16
**Fact**: The DEBI-NN C++ binary is invoked via subprocess with a project folder as its sole CLI argument. It reads `executionSettings.csv` (semicolon-separated, multi-column for ensemble members) and writes `Predictions.csv` into `Executions-Finished/{name}/Log/Fold-{N}/`. NUMA pinning via `numactl`. OMP threads via env var. Timeout: 5 days default. Qt offscreen mode required (`QT_QPA_PLATFORM=offscreen`).
**Impact**: The `debinn_.py` ModelBuilder wrapper must preserve all of this exactly.

### HF-006: Amplitude Encoding Qubit Scaling
**Date**: 2026-03-16
**Context**: EANM-AI-QC encoding.py.
**Fact**: 306 radiomics features → pad to 512 → 9 qubits. Simulation cost scales as O(2^n). The quantum kernel SVM builds an O(N²) kernel matrix where each entry is one circuit evaluation. For N=100 training samples with 9 qubits, that's ~5000 circuit evaluations for training alone.
**Impact**: Runtime warnings needed. Default sample caps for quantum models. SHAP/LIME multiply this cost by 100-1000×.

### HF-007: Benchmark Dataset Tiers
**Date**: 2026-03-16
**Fact**: 28 datasets in 4 tiers. Large dataset threshold: N > 10,000 → fixed 60/20/20 split instead of k-fold. 5 datasets exceed this threshold (electricity 45K, bank-marketing 45K, adult 49K, dry-bean 14K, mushroom 8K — last two are borderline). Threshold is configurable in config.py.

### HF-008: CLAREON Trademark Collision
**Date**: 2026-03-16
**Context**: Naming the project. CLAREON was the preferred spelling.
**Fact**: "Clareon" is Alcon's (NYSE: ALC, ~$40B market cap) flagship intraocular lens product line. Trademarked, CE-marked, FDA-cleared. Active product family: Clareon Monofocal, Clareon PanOptix, Clareon Vivity. 30+ published clinical studies. Heavily present in medical literature (PubMed).
**Decision**: Adopted Y-variant spelling **CLARYON** to avoid brand confusion in medical search results and committee presentations. Different enough to avoid trademark overlap while preserving the desired phonetic identity.

---

## 4. File Delivery Tracker

Files generated and delivered to user. Tracks what the user should have in their project.

| File | Version | Date | Status |
|---|---|---|---|
| `REQUIREMENTS.md` | v0.3.1 | 2026-03-16 | Updated with CLARYON rename, needs project drop |
| `IMPLEMENTATION_PLAN.md` | v0.1.1 | 2026-03-16 | Updated with CLARYON rename, needs project drop |
| `WORKLOG.md` | v0.2.0 | 2026-03-16 | Updated with Session 2, needs project drop |

---

## 5. Next Session Checklist

When starting the next chat:

1. Confirm REQUIREMENTS.md, IMPLEMENTATION_PLAN.md, and WORKLOG.md are in the project
2. Read WORKLOG.md Section 1 (Current State) to know where we are
3. Read WORKLOG.md Section 3 (Hard Facts) to avoid known pitfalls
4. Check IMPLEMENTATION_PLAN.md Section 7 (Session Handoff) for phase/item status
5. Begin implementation at the "Next item to build" listed in Section 1 above
