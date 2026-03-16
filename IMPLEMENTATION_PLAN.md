# CLARYON — Implementation Master Plan

**Version**: 0.1.1
**Date**: 2026-03-16
**Companion to**: REQUIREMENTS.md v0.3.1
**Purpose**: Defines *how* to build, in what order, from which sources. Serves as session handoff document.

---

## 0. Conventions

- **REQ §N.M**: Reference to REQUIREMENTS.md section N.M
- **[B]**: Source is Benchmark project
- **[E]**: Source is EANM-AI-QC project
- **[NEW]**: Written from scratch
- **PORT**: Adapt existing code into new module structure
- **REWRITE**: Logic preserved but code rewritten for new interfaces
- **SUPERSEDE**: Old file replaced entirely; not ported
- **Status tags**: `TODO`, `IN-PROGRESS`, `DONE`, `BLOCKED-BY <item>`

---

## 1. Target Project Structure

```
claryon/
├── pyproject.toml
├── README.md
├── REQUIREMENTS.md
├── IMPLEMENTATION_PLAN.md          ← this file
├── Dockerfile
├── Dockerfile.gpu
├── singularity.def
├── configs/
│   ├── example_tabular.yaml
│   ├── example_nifti.yaml
│   ├── example_benchmark.yaml
│   └── pyradiomics_default.yaml
├── claryon/
│   ├── __init__.py
│   ├── cli.py                      # CLI entry point
│   ├── pipeline.py                 # Stage orchestrator
│   ├── registry.py                 # Decorator-based plugin registry
│   ├── config_schema.py            # Pydantic config model + validation
│   ├── determinism.py              # Seed + thread control
│   ├── io/
│   │   ├── __init__.py
│   │   ├── base.py                 # Dataset dataclass + unified contract
│   │   ├── tabular.py              # CSV/Parquet loader
│   │   ├── nifti.py                # NIfTI loader (nibabel)
│   │   ├── tiff.py                 # TIFF + metadata loader
│   │   ├── fdb_ldb.py              # DEBI-NN legacy format compat
│   │   ├── dicom.py                # DICOM loader (future stub)
│   │   └── predictions.py          # Unified Predictions.csv writer/reader (REQ §8.4)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── tabular_prep.py         # Imputation, encoding, scaling
│   │   ├── image_prep.py           # Resampling, normalization, augmentation
│   │   ├── radiomics.py            # PyRadiomics wrapper + merger
│   │   └── splits.py               # k-fold, nested CV, holdout, GroupKFold
│   ├── encoding/
│   │   ├── __init__.py
│   │   ├── base.py                 # QuantumEncoding base class
│   │   ├── amplitude.py            # Amplitude encoding (port)
│   │   ├── angle.py                # Angle encoding (new)
│   │   └── iqp.py                  # IQP encoding (new, stub)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                 # ModelBuilder abstract base + task type enum
│   │   ├── classical/
│   │   │   ├── __init__.py
│   │   │   ├── xgboost_.py
│   │   │   ├── lightgbm_.py
│   │   │   ├── catboost_.py
│   │   │   ├── tabpfn_.py
│   │   │   ├── mlp_.py
│   │   │   ├── tabm_.py            # stub
│   │   │   ├── realmlp_.py         # stub
│   │   │   ├── modernnca_.py       # stub
│   │   │   ├── debinn_.py          # C++ subprocess wrapper
│   │   │   ├── cnn_2d.py           # PyTorch 2D CNN
│   │   │   └── cnn_3d.py           # MONAI/PyTorch 3D CNN
│   │   ├── quantum/
│   │   │   ├── __init__.py
│   │   │   ├── kernel_svm.py       # Quantum kernel SVM (port)
│   │   │   ├── qcnn_muw.py         # QCNN MUW variant (port)
│   │   │   ├── qcnn_alt.py         # QCNN ALT variant (port)
│   │   │   ├── vqc.py              # Variational quantum classifier (new, stub)
│   │   │   └── hybrid.py           # Hybrid quantum-classical (new, stub)
│   │   └── ensemble.py             # Softmax averaging (classification) / mean (regression)
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── base.py                 # Explainer base class
│   │   ├── shap_.py                # SHAP wrapper (port + generalize)
│   │   ├── lime_.py                # LIME wrapper (port + generalize)
│   │   ├── gradcam.py              # GradCAM for CNNs (new, stub)
│   │   ├── integrated_gradients.py # IG (new, stub)
│   │   ├── quantum_gradients.py    # Parameter-shift attribution (new, stub)
│   │   ├── conformal.py            # Conformal prediction (new, stub)
│   │   └── utils.py                # Feature selection for reduced-space explainability
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Metric registry + implementations
│   │   ├── comparator.py           # Friedman/Nemenyi, DeLong, McNemar, bootstrap CI
│   │   ├── results_store.py        # Results table builder
│   │   └── figures.py              # Publication-quality figure generators
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── latex_report.py         # Jinja2 → .tex
│   │   ├── markdown_report.py      # Jinja2 → .md
│   │   └── templates/
│   │       ├── methods.tex.j2
│   │       ├── results.tex.j2
│   │       └── full_manuscript.tex.j2
│   └── benchmark/
│       ├── __init__.py
│       ├── downloader.py           # Dataset downloader (port)
│       ├── preprocessor.py         # Benchmark preprocessing (port)
│       └── settings_generator.py   # DEBI-NN settings CSV (port)
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── fixtures/                   # Synthetic test data
│   │   └── generate_fixtures.py
│   ├── test_io/
│   ├── test_encoding/
│   ├── test_models/
│   ├── test_evaluation/
│   ├── test_explainability/
│   └── test_integration/
├── docs/
│   ├── user_guide.md
│   ├── contributor_guide.md
│   ├── config_reference.md
│   └── architecture.md
└── examples/
    ├── configs/
    │   ├── pet_ct_radiomics.yaml
    │   ├── oct_classification.yaml
    │   └── quantum_benchmark.yaml
    └── notebooks/
        ├── 00_quickstart.ipynb
        ├── 01_tabular_demo.ipynb
        ├── 02_nifti_demo.ipynb
        ├── 03_explainability.ipynb
        └── 04_results_dashboard.ipynb
```

---

## 2. Source File Disposition

### 2.1 Benchmark Project Files → Target

| Source File | Disposition | Target | Notes |
|---|---|---|---|
| `config.py` [B] | REWRITE | `claryon/config_schema.py` | Dataset registry, metric lists, paths → Pydantic config model. Constants absorbed into YAML defaults. |
| `run_benchmark.py` [B] | REWRITE | `claryon/pipeline.py` | 6-stage orchestrator logic preserved; generalized beyond DEBI-NN benchmarking. |
| `download_benchmark_datasets.py` [B] | PORT | `claryon/benchmark/downloader.py` | Direct port. Add download progress, retry logic. |
| `preprocess_benchmark.py` [B] | PORT | `claryon/preprocessing/tabular_prep.py` | Quantile normalization, one-hot encoding, imputation, missing indicators. Generalize beyond FDB/LDB format. |
| `fold_generator.py` [B] | PORT | `claryon/preprocessing/splits.py` | Stratified k-fold + large-dataset fixed split. Add nested CV, GroupKFold. |
| `project_builder.py` [B] | PORT | `claryon/benchmark/settings_generator.py` | DEBI-NN-specific project folder builder. Keep for DEBI-NN compat. |
| `settings_generator.py` [B] | PORT | `claryon/benchmark/settings_generator.py` | Merge with project_builder into single DEBI-NN support module. |
| `debinn_runner.py` [B] | PORT | `claryon/models/classical/debinn_.py` | Subprocess wrapper + NUMA pinning. Wrap in ModelBuilder interface. |
| `competitor_runner.py` [B] | REWRITE | `claryon/models/classical/*.py` | Each competitor → separate file with @register decorator. Dispatcher pattern replaced by registry. |
| `ensemble_aggregator.py` [B] | PORT | `claryon/models/ensemble.py` | Softmax averaging preserved. Add regression mean. Add task-type dispatch. |
| `results_collector.py` [B] | REWRITE | `claryon/evaluation/results_store.py` | Generalize beyond DEBI-NN paths. Consume unified Predictions.csv format. |
| `split_train_test.py` [B] | SUPERSEDE | `claryon/preprocessing/splits.py` | Functionality absorbed into general splitter. |
| `analysis.py` [B] | PORT | `claryon/evaluation/comparator.py` + `figures.py` | Friedman/Nemenyi + LaTeX table generation. Split statistics from visualization. |
| `base_settings.csv` [B] | KEEP | `configs/debinn_base_settings.csv` | DEBI-NN template. Unchanged. |
| `benchmark_log.txt` [B] | DROP | — | Runtime artifact, not source code. |
| `competitor_log.txt` [B] | DROP | — | Runtime artifact, not source code. |

### 2.2 EANM-AI-QC Project Files → Target

| Source File | Disposition | Target | Notes |
|---|---|---|---|
| `__init__.py` (root) [E] | REWRITE | `claryon/__init__.py` | Version string. Add package-level imports. |
| `io/__init__.py` [E] | REWRITE | `claryon/io/__init__.py` | Re-exports for new module structure. |
| `models/__init__.py` [E] | SUPERSEDE | `claryon/models/__init__.py` | Registry auto-discovery replaces manual re-exports. |
| `explain/__init__.py` [E] | SUPERSEDE | `claryon/explainability/__init__.py` | Registry auto-discovery. |
| `qnm_qai.py` [E] | SUPERSEDE | `claryon/cli.py` | New CLI with full subcommands. BLAS thread control moved to determinism.py. |
| `cli.py` [E] | REWRITE | `claryon/cli.py` | Expand from single `run` command to multi-stage CLI (REQ §13.2). |
| `determinism.py` [E] | PORT | `claryon/determinism.py` | Direct port. Add PyTorch seeding. |
| `common.py` [E] | REWRITE | `claryon/io/base.py` | BinaryLabelMapper → generalize to MultiClassLabelMapper + RegressionTarget. |
| `tabular.py` [E] | PORT | `claryon/io/tabular.py` | Preserve TabularDataset. Remove amplitude encoding coupling (encoding is separate module now). |
| `nifti.py` [E] | PORT | `claryon/io/nifti.py` | Preserve mask pairing logic. Remove amplitude encoding coupling. |
| `encoding.py` [E] | PORT | `claryon/encoding/amplitude.py` | Move to encoding subpackage. Extract base class. |
| `base.py` [E] | REWRITE | `claryon/models/base.py` | Expand Protocol to full abstract base class: add `task_type`, `predict()` (for regression), `load()`, `explain()` (optional). |
| `pl_kernel_svm.py` [E] | PORT | `claryon/models/quantum/kernel_svm.py` | Add @register decorator. Generalize to multi-class (one-vs-rest wrapper). |
| `pl_qcnn_muw.py` [E] | PORT | `claryon/models/quantum/qcnn_muw.py` | Add @register decorator. Decouple encoding. Generalize output to multi-class. |
| `pl_qcnn_alt.py` [E] | PORT | `claryon/models/quantum/qcnn_alt.py` | Add @register decorator. Decouple encoding. Generalize output to multi-class. |
| `runner.py` [E] | REWRITE | `claryon/pipeline.py` | Major rewrite. Merge with benchmark orchestrator. Support multi-model, multi-fold, multi-seed. Consume config schema. |
| `metrics.py` [E] | PORT | `claryon/evaluation/metrics.py` | Preserve binary_metrics + threshold optimizer. Add multi-class + regression metrics. Add @register decorator per metric. |
| `shap_explain.py` [E] | PORT | `claryon/explainability/shap_.py` | Generalize beyond binary. Keep reduced-space pattern. |
| `lime_explain.py` [E] | PORT | `claryon/explainability/lime_.py` | Generalize beyond binary. Keep reduced-space pattern. |
| `utils.py` [E] | PORT | `claryon/explainability/utils.py` | Direct port. Feature variance selection. |
| `make_synthetic_nifti.py` [E] | PORT | `tests/fixtures/generate_fixtures.py` | Absorb into test fixture generator. Add tabular + TIFF fixtures. |
| `build_tabular_from_fdb_ldb.py` [E] | PORT | `claryon/io/fdb_ldb.py` | FDB/LDB → tabular converter as a loader. |
| `run_explain_all.py` [E] | SUPERSEDE | `claryon/cli.py explain` | Functionality becomes `eanm-ai explain --config ...` subcommand. |
| `run_all_examples.sh` [E] | REWRITE | `examples/run_demo.sh` | Updated for new CLI. |
| `run_explain_all.sh` [E] | SUPERSEDE | — | Absorbed into CLI. |
| `requirements.txt` [E] | SUPERSEDE | `pyproject.toml` | Dependency groups in pyproject.toml. |
| `_gitignore` [E] | PORT | `.gitignore` | Update paths. |
| `README.md` [E] | REWRITE | `README.md` | New structure per REQ §16.3. |
| Notebooks (00–04) [E] | REWRITE | `examples/notebooks/` | Update for new API, CLI, config structure. |
| Demo CSVs [E] | KEEP | `examples/data/` | FDB.csv, LDB.csv, real_train.csv, real_infer.csv, real_feature_map.csv |

---

## 3. Dependency Chains

These define hard build order. An arrow means "must exist before".

```
registry.py
  → models/base.py
    → every model file
  → evaluation/metrics.py
    → every metric function
  → explainability/base.py
    → every explainer
  → encoding/base.py
    → every encoding

config_schema.py
  → cli.py
  → pipeline.py

io/base.py (Dataset dataclass, LabelMapper)
  → io/tabular.py
  → io/nifti.py
  → io/tiff.py
  → io/fdb_ldb.py
  → io/predictions.py

io/predictions.py (unified Predictions.csv writer/reader)
  → every model runner
  → evaluation/results_store.py

encoding/base.py
  → encoding/amplitude.py
    → models/quantum/kernel_svm.py
    → models/quantum/qcnn_muw.py
    → models/quantum/qcnn_alt.py

preprocessing/splits.py
  → pipeline.py (needs splits before any model trains)

models/base.py
  → models/classical/*.py
  → models/quantum/*.py
  → models/ensemble.py

evaluation/metrics.py
  → evaluation/results_store.py
    → evaluation/comparator.py
      → evaluation/figures.py
        → reporting/latex_report.py
```

---

## 4. Phase Gates

### Phase 0 → Phase 1 Gate

**Must be DONE before Phase 1 starts:**

| Item | Deliverable | Verification |
|---|---|---|
| `pyproject.toml` | Package definition with all dependency groups | `pip install -e .[core]` succeeds |
| `registry.py` | Decorator-based registry with `register()` and `get()` | Unit test: register + retrieve a dummy class |
| `models/base.py` | `ModelBuilder` ABC with task type enum | Unit test: subclass instantiation |
| `io/base.py` | `Dataset` dataclass, `LabelMapper` (binary + multi-class + regression) | Unit test: create, transform, inverse |
| `io/predictions.py` | `write_predictions()` and `read_predictions()` for classification + regression | Unit test: round-trip write/read, verify format matches REQ §8.4 |
| `encoding/base.py` | `QuantumEncoding` ABC | Unit test: subclass instantiation |
| `config_schema.py` | Pydantic model for experiment config | Unit test: validate example YAML, reject invalid YAML |
| `cli.py` | Skeleton with subcommands (run, preprocess, train, evaluate, explain, report) | `eanm-ai --help` works |
| `determinism.py` | Seed + thread control | Unit test: verify RNG state after seeding |
| `tests/fixtures/` | Synthetic data fixtures (tabular binary/multi/regression, NIfTI, TIFF placeholder) | Fixtures generate deterministically |
| Dependency matrix | Documented Python version + package compatibility | Table in docs or README |

### Phase 1 → Phase 2 Gate

**Must be DONE:**

| Item | Deliverable | Verification |
|---|---|---|
| `io/tabular.py` | Loads CSV/Parquet → Dataset | Unit test: load demo data, verify shapes |
| `io/nifti.py` | Loads NIfTI + mask → Dataset | Unit test: load synthetic NIfTI, verify mask application |
| `io/tiff.py` | Loads TIFF + metadata → Dataset | Unit test: load synthetic TIFF |
| `io/fdb_ldb.py` | Loads FDB/LDB → Dataset | Unit test: load demo FDB/LDB |
| `preprocessing/tabular_prep.py` | Imputation, encoding, scaling pipeline | Unit test: transform → inverse transform consistency |
| `preprocessing/radiomics.py` | PyRadiomics wrapper (feature extraction + merge) | Unit test: extract from synthetic NIfTI + merge with tabular |
| `preprocessing/splits.py` | k-fold, holdout, nested CV, GroupKFold | Unit test: verify stratification, fold sizes, reproducibility |
| `encoding/amplitude.py` | Amplitude encoding (ported) | Unit test: pad + normalize, verify quantum state validity |

### Phase 2 → Phase 3 Gate

**Must be DONE:**

| Item | Deliverable | Verification |
|---|---|---|
| All classical tabular models | XGBoost, LightGBM, CatBoost, TabPFN, MLP registered | Smoke test each: fit + predict_proba on synthetic data |
| `models/classical/debinn_.py` | DEBI-NN subprocess wrapper with ModelBuilder interface | Unit test: mock binary, verify invocation + output parsing |
| `models/ensemble.py` | Softmax averaging (classification) + mean (regression) | Unit test: 3 dummy models → ensemble prediction |
| `pipeline.py` | End-to-end: config → load → split → train → predict → evaluate (classical only) | Integration test on synthetic tabular data |

### Phase 3 → Phase 4 Gate

**Must be DONE:**

| Item | Deliverable | Verification |
|---|---|---|
| All quantum models | kernel_svm, qcnn_muw, qcnn_alt registered | Smoke test each on tiny synthetic data (≤4 qubits) |
| `encoding/amplitude.py` | Decoupled from model, usable via registry | Unit test: encode → decode consistency |
| `encoding/angle.py` | Angle encoding registered | Unit test: correct qubit count |
| Quantum models in pipeline | Config → load → encode → train quantum → predict → evaluate | Integration test: 1 quantum model on synthetic tabular |

### Phase 4 → Phase 5 Gate

**Must be DONE:**

| Item | Deliverable | Verification |
|---|---|---|
| `models/classical/cnn_2d.py` | 2D CNN registered, trains on 2D image data | Smoke test on synthetic 2D images |
| `models/classical/cnn_3d.py` | 3D CNN registered, trains on 3D NIfTI | Smoke test on synthetic NIfTI |
| Late fusion pipeline | Tabular + imaging → separate models → combined prediction | Integration test: tabular + NIfTI → ensemble |

### Phase 5 → Phase 6 Gate

**Must be DONE:**

| Item | Deliverable | Verification |
|---|---|---|
| `explainability/shap_.py` | SHAP for tabular + imaging models (binary + multi-class) | Integration test: SHAP on 1 classical + 1 quantum model |
| `explainability/lime_.py` | LIME for tabular models | Integration test: LIME output CSV generated |
| GradCAM stub | Registered, returns placeholder | Smoke test: no crash on CNN model |

### Phase 6 → Phase 7 Gate

**Must be DONE:**

| Item | Deliverable | Verification |
|---|---|---|
| `evaluation/metrics.py` | All REQ §8.2 metrics registered | Unit test each metric on known inputs |
| `evaluation/comparator.py` | Friedman/Nemenyi, DeLong, bootstrap CI | Unit test: known statistical results reproduced |
| `evaluation/figures.py` | ROC, confusion matrix, CD diagram, SHAP summary | Visual inspection on synthetic results |
| `reporting/latex_report.py` | Jinja2 → .tex with methods + results + figures | Integration test: generate .tex from synthetic experiment |
| `reporting/markdown_report.py` | Jinja2 → .md fallback | Integration test: generate .md |

---

## 5. Implementation Priority Within Each Phase

### Phase 0 — Build Order

```
1. pyproject.toml                    (unlocks pip install)
2. claryon/__init__.py           (package exists)
3. claryon/registry.py           (unlocks all registrations)
4. claryon/determinism.py        (PORT from [E], quick win)
5. claryon/io/base.py            (Dataset + LabelMapper)
6. claryon/io/predictions.py     (prediction output contract)
7. claryon/encoding/base.py      (quantum encoding ABC)
8. claryon/models/base.py        (ModelBuilder ABC)
9. claryon/explainability/base.py (Explainer ABC)
10. claryon/config_schema.py     (Pydantic config)
11. claryon/cli.py               (skeleton)
12. claryon/pipeline.py          (skeleton — stage stubs)
13. tests/fixtures/                  (synthetic data generators)
14. tests/conftest.py                (shared fixtures)
15. Phase 0 unit tests
16. Dependency compatibility matrix
```

### Phase 1 — Build Order

```
1. claryon/io/tabular.py         (PORT from [E], decouple encoding)
2. claryon/io/nifti.py           (PORT from [E], decouple encoding)
3. claryon/io/tiff.py            (NEW, basic TIFF loader)
4. claryon/io/fdb_ldb.py         (PORT from [E]+[B], FDB/LDB compat)
5. claryon/encoding/amplitude.py (PORT from [E])
6. claryon/preprocessing/tabular_prep.py (PORT from [B] preprocess_benchmark.py)
7. claryon/preprocessing/splits.py (PORT from [B] fold_generator.py + add nested CV)
8. claryon/preprocessing/radiomics.py (NEW, pyradiomics wrapper)
9. claryon/preprocessing/image_prep.py (NEW, basic resampling/norm)
10. Phase 1 unit tests
```

### Phase 2 — Build Order

```
1. claryon/models/classical/xgboost_.py   (REWRITE from [B])
2. claryon/models/classical/lightgbm_.py  (REWRITE from [B])
3. claryon/models/classical/catboost_.py   (REWRITE from [B])
4. claryon/models/classical/mlp_.py        (REWRITE from [B])
5. claryon/models/classical/tabpfn_.py     (REWRITE from [B])
6. claryon/models/classical/debinn_.py     (PORT from [B])
7. claryon/models/classical/tabm_.py       (stub)
8. claryon/models/classical/realmlp_.py    (stub)
9. claryon/models/classical/modernnca_.py  (stub)
10. claryon/models/ensemble.py             (PORT from [B] + extend)
11. claryon/pipeline.py                    (flesh out stages 1-4 for classical)
12. Phase 2 smoke tests + integration test
```

### Phase 3 — Build Order

```
1. claryon/encoding/angle.py               (NEW)
2. claryon/models/quantum/kernel_svm.py    (PORT from [E])
3. claryon/models/quantum/qcnn_muw.py      (PORT from [E])
4. claryon/models/quantum/qcnn_alt.py      (PORT from [E])
5. claryon/models/quantum/vqc.py           (NEW, stub)
6. claryon/models/quantum/hybrid.py        (NEW, stub)
7. Pipeline integration for quantum models
8. Phase 3 smoke tests + integration test
```

### Phase 4 — Build Order

```
1. claryon/models/classical/cnn_2d.py      (NEW)
2. claryon/models/classical/cnn_3d.py      (NEW)
3. Late fusion in pipeline.py
4. claryon/preprocessing/image_prep.py     (extend: augmentation, MONAI transforms)
5. Phase 4 smoke tests + integration test
```

### Phase 5 — Build Order

```
1. claryon/explainability/shap_.py         (PORT from [E], generalize)
2. claryon/explainability/lime_.py         (PORT from [E], generalize)
3. claryon/explainability/utils.py         (PORT from [E])
4. claryon/explainability/gradcam.py       (NEW, stub)
5. claryon/explainability/integrated_gradients.py (NEW, stub)
6. claryon/explainability/quantum_gradients.py    (NEW, stub)
7. claryon/explainability/conformal.py            (NEW, stub)
8. Phase 5 tests
```

### Phase 6 — Build Order

```
1. claryon/evaluation/metrics.py           (PORT from [E]+[B], registry)
2. claryon/evaluation/comparator.py        (PORT from [B] analysis.py)
3. claryon/evaluation/results_store.py     (REWRITE from [B])
4. claryon/evaluation/figures.py           (NEW + PORT from [B])
5. claryon/reporting/latex_report.py       (NEW, Jinja2)
6. claryon/reporting/markdown_report.py    (NEW, Jinja2)
7. claryon/reporting/templates/*.tex.j2    (NEW)
8. Phase 6 tests
```

### Phase 7 — Build Order

```
1. GitHub Actions CI workflow
2. Dockerfile + Dockerfile.gpu
3. singularity.def
4. Full integration test suite
5. Benchmark regression test suite
6. API docs (Sphinx/MkDocs config + docstring audit)
7. User guide
8. Model contributor guide
9. Config reference (auto-generated from Pydantic schema)
10. README rewrite
11. examples/notebooks/ rewrite
```

---

## 6. Cross-Cutting Concerns (Apply to Every Phase)

| Concern | Rule |
|---|---|
| **Docstrings** | Every public function/class gets a Google-style docstring at creation time, not retroactively. |
| **Type hints** | All function signatures fully typed. `from __future__ import annotations` in every file. |
| **Registry decoration** | Every model, metric, explainer, and encoding is registered at creation time. |
| **Prediction contract** | All model outputs go through `io/predictions.py` — models never write CSVs directly. |
| **Semicolon separator** | All CSV I/O uses `;` separator (REQ §8.4). Internal data is pandas/numpy; serialization is centralized. |
| **Seed propagation** | Every stochastic operation receives the experiment seed or a derived seed. No unseeded randomness. |
| **Logging** | Use Python `logging` module, not `print()`. Configurable verbosity. |

---

## 7. Session Handoff Protocol

At the end of each implementation session, update this section:

### Current Status

- **Phase**: 0 (not started)
- **Last completed item**: —
- **Next item to build**: `pyproject.toml`
- **Blockers**: None
- **Open questions**: None

### Change Log

| Date | Phase | Items Completed |
|---|---|---|
| 2026-03-16 | — | Requirements v0.3.0 + Implementation Plan v0.1.0 created |
