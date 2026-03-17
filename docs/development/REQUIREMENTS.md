# CLARYON — Requirements Specification

**Version**: 0.3.1
**Date**: 2026-03-16
**Authors**: L. Papp (MedUni Wien / ), Claude (Anthropic, scaffolding assistant)
**Status**: DRAFT — revision 3.1, renamed project to CLARYON

---

## 1. Project Identity

### 1.1 Name & Scope

**CLARYON** — **CL**assical-quantum **A**I for **R**eproducible Explainable **O**pe**N**-source medicine.

**Package name**: `claryon` (`pip install claryon`)

**Two-repo strategy**:
- **CLARYON repo** (`github.com/<org>/claryon`): All code, tests, CI, documentation. The engineering artifact.
- **EANM-AI-QC repo** (existing): Educational hub for the EANM AI Committee — teaching materials, slide decks, tutorial walkthroughs, committee documentation. Links to CLARYON for the actual code. No Python code.

The CLARYON codebase absorbs functionality from two existing Python projects:

1. **EANM-AI-QC** (v0.8.0): quantum ML framework for nuclear medicine (PennyLane QCNN, quantum kernel SVM, SHAP/LIME explainability, NIfTI + tabular I/O).
2. **DEBI-NN Benchmark Harness**: tabular classification benchmark across 28 datasets with 8 classical competitor methods, statistical analysis, and LaTeX reporting.

The combined framework must also support non-nuclear-medicine domains including optical coherence tomography (OCT), photoacoustics, biophotonics, and general medical imaging research.

**Target communities**: EANM AI Committee, OCT research groups (MedUni Wien and external), quantum AI researchers.

### 1.2 License

Open-source research license. Exact license TBD. Must be compatible with academic research use, community contributions, and integration with GPLv2/v3-licensed dependencies (e.g., pyradiomics).

### 1.3 Python Version

To be determined based on dependency compatibility analysis. Candidates: Python 3.10–3.12. The version floor will be selected to maximize compatibility across PennyLane, MONAI, pyradiomics, and other heavy dependencies. A compatibility matrix will be produced during Phase 0.

---

## 2. Task Types (Classification Scope)

### 2.1 Supervised Learning Tasks

The framework must support:

| Task Type | Output Contract | Primary Metrics | Status |
|---|---|---|---|
| **Binary classification** | P(class=1) per sample | BACC, AUC, sensitivity, specificity, PPV, NPV, 95% CI | MVP |
| **Multi-class classification** | P(class=k) per sample, k=0..K-1 | BACC, macro-F1, weighted-F1, MCC, confusion matrix | MVP |
| **Regression** | Continuous predicted value per sample | MAE, RMSE, R², Spearman ρ | MVP |
| **Ordinal regression** | Ordered class probabilities | Quadratic weighted kappa, ordinal C-index | Planned |
| **Survival / time-to-event** | Risk score or hazard per sample | C-index, Brier score, calibration | Planned |

### 2.2 Quantum Model Task Support

Quantum models must support the same task types as classical models (binary, multi-class, regression). For multi-class, acceptable approaches include one-vs-rest wrappers or native multi-class quantum circuits. For regression, the quantum circuit output is interpreted as a continuous value rather than a class probability.

### 2.3 Physics-Informed AI (Future)

The architecture must not preclude physics-informed AI extensions where:

- The loss function incorporates domain-specific physical constraints (e.g., conservation laws, PDE residuals).
- The model architecture may include physics-based inductive biases.

**Requirement**: The loss function and model interface must be extensible via the registry/plugin system without modifying core pipeline code.

---

## 3. Data Modalities & I/O

### 3.1 Supported Input Formats

| Format | Domain | Loader |
|---|---|---|
| **NIfTI** (.nii, .nii.gz) | Nuclear medicine PET/CT/MR, neuroimaging | nibabel |
| **TIFF** (.tif, .tiff) + metadata | OCT, photoacoustics, biophotonics | tifffile / scikit-image |
| **Tabular CSV/Parquet** | Clinical, radiomics, any flat features | pandas |
| **DICOM** (.dcm) | General medical imaging | pydicom (planned) |
| **FDB/LDB** (semicolon-separated CSV) | DEBI-NN legacy format | pandas (compatibility wrapper) |

### 3.2 Radiomics Extraction

The framework must support two radiomics workflows:

1. **Pre-extracted radiomics**: User provides a tabular CSV of already-extracted features. The framework loads and uses them directly.
2. **Integrated extraction**: User provides NIfTI image + mask pairs. The framework invokes pyradiomics internally to extract features, then merges them with any existing tabular data for the same patient (joined on patient/subject ID).

PyRadiomics configuration (feature classes, bin width, resampling, etc.) must be specifiable via a YAML/JSON config file passed to the extraction module.

### 3.3 Multi-Modal Fusion

When a patient has both imaging data and tabular data, the framework supports:

| Strategy | Description | Status |
|---|---|---|
| **Early fusion (flatten)** | Flatten image features (e.g., radiomics, CNN embeddings, or raw voxels) → concatenate with tabular features → single model. Available as a baseline option. Architecturally simple but may be suboptimal for high-dimensional raw imaging data; best suited when the imaging side is already reduced to a feature vector (radiomics, bottleneck embeddings). | MVP |
| **Late fusion** | Separate models for image and tabular → combine predictions (e.g., averaging, learned weighting) | MVP |
| **Intermediate fusion** | Shared embedding space where image encoder and tabular encoder produce compatible representations that are merged before a joint prediction head | Planned |

### 3.4 Mask Semantics

For imaging data with accompanying masks:

- Masks define the region of interest (ROI) for feature extraction (radiomics) or model input (CNN).
- Mask format: same spatial dimensions as the image, binary or integer-labelled.
- Multi-label masks (e.g., multiple lesions) must be supported — each label treated as a separate ROI for radiomics extraction.

---

## 4. Quantum Computing Scope

### 4.1 Backend

**Primary**: PennyLane (`default.qubit` simulator). This is the current implementation and the most widely used quantum ML framework in the research community.

**Secondary (planned)**: Qiskit. The architecture must support adding Qiskit-based models without modifying the core pipeline. Backend selection is a model-level concern, not a pipeline-level concern.

**Decision**: Start with PennyLane only. Extend to Qiskit when a concrete use case requires it. The model registry pattern accommodates this naturally.

### 4.2 Qubit Constraints

All quantum computation runs on classical simulators. Practical qubit limit: **≤30 qubits**. State vector simulation cost scales as O(2^n); runs above ~20 qubits should be flagged as long-running with estimated runtime warnings.

### 4.3 Quantum Encoding Module

Quantum state encoding is a separate, modular concern. The framework must support multiple encoding strategies via a registry:

| Encoding | Description | Qubits Required | Status |
|---|---|---|---|
| **Amplitude encoding** | Pad to 2^n, L2-normalize | log₂(feature_count) | Implemented |
| **Angle encoding** | One feature per qubit rotation | = feature_count | Planned |
| **IQP encoding** | Feature-dependent entangling gates | = feature_count | Planned |
| **Custom** | User-defined encoding circuit | User-defined | Plugin via registry |

The encoding module must be decoupled from the model module so that any encoding can be paired with any quantum model.

### 4.4 Quantum Model Types

| Model Type | Description | Status |
|---|---|---|
| **Quantum kernel SVM** | Amplitude-kernel + classical SVC | Implemented |
| **QCNN (MUW variant)** | Conv/pool + ArbitraryUnitary head | Implemented |
| **QCNN (ALT variant)** | Conv/pool + StronglyEntanglingLayers head | Implemented |
| **VQC** | Variational quantum classifier (parameterized circuit) | Planned |
| **Hybrid quantum-classical** | Classical encoder → quantum circuit → classical decoder | Planned |
| **Quantum transfer learning** | Pretrained classical model (e.g., ResNet) → quantum fine-tuning circuit | Planned |
| **Quantum-assisted classical** | Quantum circuit output trains a classical model; circuit optimized by classical optimizer (e.g., COBYLA) | Planned |

### 4.5 Quantum-Specific Concerns

- **Decision threshold optimization**: Quantum models often produce poorly-calibrated probabilities near 0.5. The framework must retain Youden's J threshold optimization on the training set (already implemented).
- **Barren plateau detection**: Log gradient norms during training; warn if gradients vanish.
- **Noise model injection**: Planned. Realistic simulator noise profiles for benchmarking quantum model robustness.

---

## 5. Classical Model Scope

### 5.1 Tabular Models (MVP)

| Model | Library | Status |
|---|---|---|
| **XGBoost** | xgboost | Implemented (benchmark) |
| **LightGBM** | lightgbm | Implemented (benchmark) |
| **CatBoost** | catboost | Implemented (benchmark) |
| **TabPFN** | tabpfn | Implemented (benchmark) |
| **MLP** | scikit-learn | Implemented (benchmark) |
| **DEBI-NN** | C++ binary (external) | Implemented (benchmark) |
| **TabM** | yandex-research/tabm | Stub (benchmark) |
| **RealMLP** | pytabkit | Stub (benchmark) |
| **ModernNCA** | TALENT repo | Stub (benchmark) |

### 5.2 Imaging Models (MVP)

| Model | Library | Data Type | Status |
|---|---|---|---|
| **2D CNN** | torchvision / PyTorch | 2D slices (OCT, etc.) | Planned |
| **3D CNN** | MONAI or PyTorch | 3D volumes (PET/CT/MR) | Planned |
| **ResNet (pretrained)** | torchvision / MONAI | Transfer learning base | Planned |

### 5.3 MONAI vs. PyTorch

MONAI provides substantial medical imaging infrastructure (transforms, losses, pretrained models) but is a heavy dependency with potential conflicts.

**Strategy**:

- MONAI is an optional dependency group.
- The framework must function without MONAI, falling back to raw PyTorch for imaging models.
- **MVP approach**: Dependency groups are documented as mutually compatible or incompatible. The user installs the groups they need. If incompatible libraries are both required for a study, the user runs them in separate virtual environments consuming the same pre-generated CV splits (splits are stored as CSV files on disk, environment-independent).
- **Future**: Automated subprocess-based batch execution for incompatible module groups within a single invocation.

### 5.4 Model Extensibility

Any new model (classical or quantum, tabular or imaging) must be addable by:

1. Creating a single Python file implementing the `ModelBuilder` interface.
2. Decorating the class with `@register("model", "model_name")`.
3. No modifications to pipeline, CLI, or evaluation code.

The `ModelBuilder` interface enforces: `fit()`, `predict()`, `predict_proba()` (for classification), `save()`, `load()`, and `explain()` (optional, for model-specific explainability).

---

## 6. Cross-Validation & Evaluation Protocol

### 6.1 Default Protocol

**Default**: Stratified k-fold cross-validation (k=5, configurable). Multiple seeds for variance estimation.

**Large datasets** (N > configurable threshold): Fixed stratified split (60/20/20 train/val/test).

**Quantum models**: Same CV protocol as classical models. Runtime warnings are issued when estimated fold execution time exceeds a configurable threshold (e.g., 1 hour per fold). User may opt for reduced-fold (e.g., k=3) or single-split evaluation for quantum models while retaining k-fold for classical models within the same experiment.

### 6.2 Configurable Alternatives

| Protocol | Use Case | Config |
|---|---|---|
| **k-fold CV** | Default, preferred | `cv_strategy: kfold, n_folds: 5` |
| **Holdout** | Quick iteration, quantum models | `cv_strategy: holdout, test_size: 0.2` |
| **Nested CV** | Hyperparameter tuning with unbiased estimation | `cv_strategy: nested, outer_folds: 5, inner_folds: 3` |
| **External test set** | When independent test data exists | `cv_strategy: external, test_path: ...` |
| **GroupKFold** | Multi-center studies, patient-level grouping | `cv_strategy: group_kfold, group_col: center_id` |

### 6.3 Split Consistency

**Critical requirement**: The cross-validation splits must be global across all AI models for a given experiment. All models (classical, quantum, imaging, tabular) evaluate on exactly the same train/test partitions for fair comparison. Splits are generated once and stored; all model runners consume the stored splits.

### 6.4 Ensemble

**Classification (binary and multi-class)**: Softmax probability averaging across ensemble members. All classification model outputs are normalized to softmax probabilities before averaging. Argmax of the averaged probabilities produces the ensemble prediction.

**Regression**: Raw prediction averaging across ensemble members. No softmax normalization — regression outputs are continuous values averaged directly. Ensemble prediction is the arithmetic mean of member predictions.

**Implementation note**: The ensemble aggregator must check the task type (classification vs. regression) from the experiment config and apply the correct aggregation strategy. Mixing classification and regression models in a single ensemble is not permitted.

**Not yet in scope**: Stacking, blending, learned ensemble weights. These can be added later via the model registry (an ensemble method is itself a registered model that consumes other models' outputs).

---

## 7. Explainability

### 7.1 Model-Agnostic Explainers

| Explainer | Scope | Status |
|---|---|---|
| **SHAP** (PermutationExplainer) | All models (tabular and imaging via flattened features) | Implemented |
| **LIME** (LimeTabularExplainer) | Tabular models and feature-vector models | Implemented |
| **Integrated Gradients** | Differentiable models (CNNs, QCNNs via parameter-shift) | Planned |
| **Conformal Prediction** | Distribution-free uncertainty intervals for all models | Planned |

### 7.2 Model-Specific Explainers

| Explainer | Model Type | Status |
|---|---|---|
| **GradCAM / Grad-CAM++** | Classical CNNs (2D/3D) | Planned |
| **Attention maps** | Transformer-based models | Planned |
| **Quantum parameter-shift gradients** | Quantum circuits (PennyLane) | Planned |

### 7.3 Explainability Output

**Dual output**:

1. **Raw data**: SHAP values as NPY/CSV, LIME weights as CSV, feature importance rankings, per-sample attributions. These are the primary artifacts for downstream analysis.
2. **Publication-quality figures**: Auto-generated matplotlib/seaborn figures (SHAP beeswarm, LIME bar plots, GradCAM overlays, feature importance rankings, ROC curves with CI). Figures saved as PDF and PNG at 300 DPI.

### 7.4 Reduced-Space Explainability

For high-dimensional data (e.g., radiomics with 300+ features, or flattened imaging data), SHAP/LIME operate in a reduced feature space (top-K features by variance). The model is always evaluated on the full feature vector via baseline expansion. This pattern (already implemented in EANM-AI-QC) must be preserved and generalized.

---

## 8. Performance Evaluation & Metrics

### 8.1 Metric Registry

The metric module must be a separate, easily extensible component. Metrics within the EANM community are an active research question — the module will evolve.

**Extension mechanism**: Adding a new metric requires one function with a decorator `@register("metric", "metric_name")`, specifying whether higher-is-better and which task types it applies to.

### 8.2 Core Metrics

**Classification (binary)**:

- Confusion matrix entries (TP, TN, FP, FN)
- Balanced accuracy (BACC)
- AUC (ROC)
- Sensitivity, specificity, PPV, NPV
- 95% confidence intervals (bootstrap)
- p-values where relevant (DeLong test for AUC comparison, McNemar test)

**Classification (multi-class)**:

- Per-class confusion matrix
- Balanced accuracy, macro-F1, weighted-F1
- MCC (Matthews correlation coefficient)
- Per-class AUC (one-vs-rest)
- Entropy loss (cross-entropy)

**Regression**:

- MAE, RMSE, R², Spearman ρ, Pearson r
- Residual plots

### 8.3 Statistical Comparison

- **Friedman test + Nemenyi post-hoc** (ranking across datasets)
- **DeLong test** (pairwise AUC comparison)
- **McNemar test** (pairwise classification comparison)
- **Bootstrap 95% CI** (per-metric confidence intervals)
- **Critical difference diagrams** (visualization of Nemenyi results)

### 8.4 Prediction Output Contract

All models produce predictions in a single, unified format. This is a foundational contract — every model, evaluator, explainer, and reporting module depends on it.

**Separator**: Semicolon (`;`) throughout the framework. This matches the DEBI-NN FDB/LDB convention and avoids conflicts with comma-containing feature names or values.

**Classification (binary and multi-class)**:

```
Key;Actual;Predicted;P0;P1;...;PK-1
```

| Column | Type | Description |
|---|---|---|
| `Key` | string | Unique sample identifier (patient/subject ID) |
| `Actual` | integer | Ground truth label (0..K-1). Empty if unavailable (inference-only data). |
| `Predicted` | integer | Predicted class label (argmax of probabilities, or threshold-based for binary) |
| `P0`..`PK-1` | float (%.8f) | Per-class predicted probabilities. Must sum to ~1.0 per row. |

**Regression**:

```
Key;Actual;Predicted
```

| Column | Type | Description |
|---|---|---|
| `Key` | string | Unique sample identifier |
| `Actual` | float | Ground truth continuous value. Empty if unavailable. |
| `Predicted` | float | Predicted continuous value |

**Additional metadata columns** (optional, appended after the core columns):

| Column | Type | Description |
|---|---|---|
| `Threshold` | float | Decision threshold used (binary classification only) |
| `Fold` | integer | CV fold index (if applicable) |
| `Seed` | integer | CV seed (if applicable) |

**File naming convention**: `Predictions.csv` per model per fold, stored at:
```
Results/{experiment}/predictions/{model}/seed_{s}/fold_{f}/Predictions.csv
```

**Contract rules**:

1. Every registered model's `predict()` or `predict_proba()` output must be convertible to this format by the pipeline — models do not write CSVs directly.
2. The `Key` column must be preserved end-to-end from data loading through prediction to evaluation. This enables patient-level tracing and multi-modal join operations.
3. Float precision: `%.8f` for probabilities and regression predictions. This matches the existing DEBI-NN convention and prevents rounding-induced metric drift.
4. Encoding: UTF-8. No BOM. No quoting unless a value contains the separator character.

This enables post-hoc re-evaluation with new metrics without retraining.

---

## 9. Reporting

### 9.1 LaTeX Report Generator

The framework auto-generates a LaTeX document containing:

| Section | Content |
|---|---|
| **Methods** | Reproducible description of data preprocessing, feature extraction, model architectures, training protocol, CV strategy, metrics used. Structured for direct inclusion in a manuscript. |
| **Results** | Performance tables (mean ± std across folds/seeds), statistical comparison tables, per-dataset breakdowns, bold-best formatting. |
| **Figures** | ROC curves, confusion matrices, SHAP/LIME summary plots, critical difference diagrams, calibration curves. Auto-inserted with captions. |
| **Full manuscript skeleton** | Introduction placeholder, Methods, Results, Discussion placeholder — structured for quick adaptation into a journal submission. |
| **Structured clinical report** | Alternative output format aimed at clinicians: plain-language summary of model performance, confidence intervals, clinical decision context. |

### 9.2 Fallback

Markdown-formatted report as fallback for users without a full TeX distribution.

---

## 10. Extensibility & Plugin Architecture

### 10.1 Registry Pattern

All major extension points use a decorator-based registry:

```python
@register("model", "my_new_model")
class MyNewModel(ModelBuilder): ...

@register("metric", "quadratic_kappa")
def quadratic_kappa(y_true, y_pred): ...

@register("explainer", "grad_cam")
class GradCAMExplainer(Explainer): ...

@register("encoding", "angle_encoding")
class AngleEncoding(QuantumEncoding): ...
```

### 10.2 Plugin System

Beyond the internal registry, the architecture supports external plugins that live in separate repos/packages and register at import time. This protects the core from bloat while enabling community contributions.

**Core vs. plugin boundary**: The "vanilla" core contains fundamental models (XGBoost, QCNN, kernel SVM) and the full pipeline. Experimental, domain-specific, or dependency-heavy models live as external plugins.

### 10.3 Contributor Path

1. **Immediate (MVP)**: Primary maintainer (L. Papp) adds models via registry. Model template provided.
2. **Near-term**: Contributor documentation, model template with example, CI enforcement (new model must pass integration test on synthetic data).
3. **Long-term**: External plugin support, community PRs, automated benchmark re-runs on new model submissions.

---

## 11. Dependency Management

### 11.1 Optional Dependency Groups

Heavy dependencies are organized into installable extras in `pyproject.toml`:

| Group | Packages | Use Case |
|---|---|---|
| `core` | numpy, pandas, scikit-learn, scipy, joblib | Always required |
| `quantum` | pennylane | Quantum models |
| `quantum-qiskit` | qiskit | Qiskit-based quantum models |
| `imaging` | nibabel, tifffile, pydicom | Medical image I/O |
| `radiomics` | pyradiomics | Feature extraction |
| `monai` | monai, torch, torchvision | 3D CNN models |
| `torch` | torch, torchvision | 2D CNN models (without MONAI) |
| `boosting` | xgboost, lightgbm, catboost | Tree-based models |
| `tabular-dl` | tabpfn, pytabkit | Deep tabular models |
| `explain` | shap, lime | Explainability |
| `report` | jinja2, matplotlib, seaborn | Reporting and figures |
| `all` | All of the above | Full installation |

### 11.2 Dependency Compatibility

A documented compatibility matrix specifies which dependency groups can be installed together. For MVP, this is documentation-only — users manage their own virtual environments.

**Known conflicts to document**:

- MONAI pins specific PyTorch versions that may conflict with other torch-dependent packages.
- PyRadiomics has C extension dependencies that may conflict on certain platforms.
- PennyLane and Qiskit both provide quantum simulation backends; co-installation is supported but tested configurations will be listed.

**CV split portability**: Because CV splits are stored as index arrays on disk (NumPy `.npy` files), a user can generate splits in one environment and consume them in another. This enables running classical models in environment A and quantum models in environment B on the same splits.

### 11.3 Python Version Selection

The Python version floor will be determined by running a compatibility matrix across all required and optional dependencies. This analysis happens in Phase 0.

---

## 12. Operational Requirements

### 12.1 Execution Environments

| Environment | Support Level |
|---|---|
| **Single workstation** (CPU only) | Full support |
| **Single workstation** (CPU + 1 GPU) | Full support (PyTorch/MONAI GPU acceleration) |
| **Multi-node HPC cluster** (SLURM/PBS) | Supported via config (job scripts, resource specs) |

Execution environment is configurable. The framework detects available hardware (CPU count, GPU availability) and adjusts defaults accordingly.

### 12.2 Reproducibility

| Mechanism | Details |
|---|---|
| **Seeding** | All RNGs (Python, NumPy, PennyLane, PyTorch, sklearn) seeded from a single master seed per experiment |
| **BLAS thread control** | OMP, MKL, OpenBLAS thread counts set to 1 for deterministic float reductions (configurable) |
| **Dependency pinning** | `requirements.lock` or `pyproject.toml` with exact versions for reproducible environments |
| **Config hashing** | Each experiment produces a hash of its full configuration for provenance tracking |

### 12.3 Experiment Tracking & Versioning

Each experiment run produces a self-contained provenance record:

| Artifact | Content | Format |
|---|---|---|
| **Config snapshot** | Complete YAML config as used (with defaults filled in) | `experiment_config.yaml` |
| **Config hash** | SHA-256 of the resolved config for deduplication | Stored in metadata |
| **Run manifest** | Timestamp, git commit (if in a repo), Python version, installed package versions, hostname, hardware summary | `run_manifest.json` |
| **CV splits** | All train/test index arrays, stored once and consumed by all models | `splits/seed_{s}/fold_{f}/train_idx.npy`, `test_idx.npy` |
| **Raw predictions** | Per model, per fold, per split: probabilities + ground truth + sample IDs | `predictions/{model}/seed_{s}/fold_{f}/test.csv` |
| **Metrics** | Per model, per fold: all computed metrics | `metrics/{model}/seed_{s}/fold_{f}/metrics.json` |
| **Model artifacts** | Weights, hyperparameters, training logs | `models/{model}/seed_{s}/fold_{f}/` |
| **Explain artifacts** | SHAP values, LIME weights, figures | `explain/{model}/seed_{s}/fold_{f}/` |
| **Summary** | Aggregated results table across all models/folds/seeds | `results_table.csv` |

**Experiment directory structure**:

```
Results/{experiment_name}_{timestamp}/
├── experiment_config.yaml
├── run_manifest.json
├── splits/
├── predictions/
├── metrics/
├── models/
├── explain/
├── figures/
├── results_table.csv
└── report/
    ├── report.tex
    └── report.md
```

**External tracking integration (future)**: The provenance record is self-contained and file-based. Future integration with MLflow, Weights & Biases, or similar platforms can be added as an optional logger that mirrors the file-based artifacts to the external system. This is not in MVP scope.

### 12.4 Benchmark Dataset Downloader

The framework includes a dataset downloader (`download_benchmark_datasets.py`, ported from the Benchmark project) that retrieves standardized benchmark datasets for reproducible evaluation:

| Source | Datasets | Mechanism |
|---|---|---|
| **OpenML** | 14 datasets (Tier 1–2) | `sklearn.datasets.fetch_openml` |
| **UCI ML Repository** | 10 datasets (Tier 3–4) | `ucimlrepo.fetch_ucirepo` |
| **Kaggle** | 4 datasets (Tier 3–4) | `kaggle` API |

Total: 28 datasets spanning binary/multi-class classification, small (150 samples) to large (48K samples), general-purpose and medical domains.

Downloaded datasets are stored in a standardized format (`features.csv` + `labels.csv` + `info.json` per dataset) and preprocessed into the framework's internal format via the preprocessing pipeline. A `MANIFEST.json` records download provenance.

Users may also register custom benchmark datasets via the config file.

### 12.5 Containerization

| Artifact | Purpose |
|---|---|
| **Dockerfile** | CPU-only reproducible environment |
| **Dockerfile.gpu** | GPU-enabled environment (CUDA + PyTorch) |
| **Singularity definition** | HPC cluster deployment (no root required) |
| **docker-compose.yml** | Multi-service setup (e.g., experiment runner + results server) |

Container images will be versioned and published alongside code releases.

---

## 13. Configuration

### 13.1 YAML-Driven Pipeline

All experiments are specified via a single YAML configuration file. Non-coders can swap models, change CV strategy, or add datasets by editing config — not code.

Example structure:

```yaml
experiment:
  name: prostate_psma_benchmark
  seed: 42
  results_dir: Results/

data:
  tabular:
    path: data/radiomics.csv
    label_col: gleason_risk
    id_col: patient_id
  imaging:
    path: data/pet_nifti/
    format: nifti
    mask_pattern: "*mask*.nii.gz"
  radiomics:
    extract: true
    config: configs/pyradiomics_default.yaml
  fusion: late

cv:
  strategy: kfold
  n_folds: 5
  seeds: [42, 123, 456]

models:
  - name: xgboost
    type: tabular
  - name: pl_qcnn_alt
    type: tabular_quantum
    params:
      epochs: 15
      lr: 0.02
  - name: resnet3d
    type: imaging
    params:
      pretrained: true

explainability:
  shap: true
  lime: true
  grad_cam: true
  max_features: 32
  max_test_samples: 5

evaluation:
  metrics: [bacc, auc, sensitivity, specificity, mcc]
  statistical_tests: [friedman_nemenyi, delong, bootstrap_ci]
  confidence_level: 0.95

reporting:
  latex: true
  markdown: true
  figures: true
  figure_dpi: 300
```

### 13.2 CLI

Command-line interface for common operations:

```bash
# Full experiment from config
eanm-ai run --config experiment.yaml

# Single stage
eanm-ai preprocess --config experiment.yaml
eanm-ai train --config experiment.yaml --model xgboost
eanm-ai evaluate --config experiment.yaml
eanm-ai explain --config experiment.yaml
eanm-ai report --config experiment.yaml

# Utilities
eanm-ai list-models
eanm-ai list-metrics
eanm-ai validate-config experiment.yaml
eanm-ai extract-radiomics --image data/pet.nii.gz --mask data/mask.nii.gz
```

---

## 14. Known External Dependencies (Non-Python)

| Dependency | Purpose | Bundled? |
|---|---|---|
| **DEBI-NN C++ binary** | Proprietary spatial AI model | No — invoked via subprocess. Path configurable. |
| **LaTeX distribution** | PDF report compilation | No — user installs. Markdown fallback provided. |
| **CUDA toolkit** | GPU acceleration for PyTorch/MONAI | No — optional, auto-detected. |

---

## 15. Testing Strategy

### 15.1 Test Data Fixtures

The framework ships synthetic data fixtures that run in seconds on any machine:

| Fixture | Content | Purpose |
|---|---|---|
| **Synthetic tabular (binary)** | 100 samples, 10 features, 2 classes | Smoke test all tabular models |
| **Synthetic tabular (multi-class)** | 200 samples, 15 features, 4 classes | Multi-class pipeline validation |
| **Synthetic tabular (regression)** | 100 samples, 10 features, continuous target | Regression pipeline validation |
| **Synthetic NIfTI** | 10 train + 6 test, 10×12×8 volumes ± masks | Imaging pipeline validation (ported from existing `make_synthetic_nifti.py`) |
| **Synthetic TIFF** | Small 2D images with metadata | OCT/photoacoustics pipeline validation |
| **Synthetic FDB/LDB** | DEBI-NN format, 50 samples | Legacy format compatibility |

Fixtures are generated deterministically (fixed seed) and stored in `tests/fixtures/`. They are regenerated by a script if missing.

### 15.2 Test Levels

| Level | Scope | Trigger | Runtime Target |
|---|---|---|---|
| **Unit tests** | Individual functions (encoding, metrics, label mapping, data loaders, split generators) | Every commit | < 30 seconds |
| **Model smoke tests** | Every registered model: `fit()` + `predict_proba()` on synthetic data. Verifies output shape, probability constraints (sum to 1, range [0,1]), and no exceptions. | Every commit | < 5 minutes |
| **Integration tests** | Full pipeline: config → load → split → train → evaluate → report on synthetic data. One classical model + one quantum model. | Every PR | < 15 minutes |
| **Benchmark regression tests** | Run a subset of benchmark datasets (3 small datasets × 2 models), verify metrics are within expected tolerance of stored baselines. | Weekly / release | < 1 hour |

### 15.3 CI Runner

GitHub Actions with a matrix:

- Python versions: minimum supported + latest stable
- OS: Ubuntu (primary), macOS (secondary)
- Dependency groups: `core` only, `core+quantum`, `core+boosting`, `all`

Quantum model smoke tests run on `core+quantum`. Classical model tests run on `core+boosting`. Full integration tests run on `all`.

### 15.4 Test Contract for New Models

Any model registered via `@register("model", ...)` must pass the smoke test automatically. The CI discovers all registered models and runs `fit()` + `predict_proba()` on the appropriate synthetic fixture (tabular or imaging, matching the model's declared input type). Failure blocks the PR.

---

## 16. Documentation Strategy

### 16.1 Documentation Artifacts

| Artifact | Tool | Content | Audience |
|---|---|---|---|
| **API reference** | Sphinx (autodoc) or MkDocs (mkdocstrings) | Auto-generated from docstrings. Every public class, function, and module documented. | Developers, contributors |
| **User guide** | Markdown in `docs/` | "How to run your first experiment" tutorial. Step-by-step with example config, expected output, and interpretation. | Clinicians, researchers new to the tool |
| **Model contributor guide** | Markdown in `docs/` | "How to add a new model" walkthrough. Template file, registry decorator, required interface methods, testing requirements, PR checklist. | Community contributors |
| **Config reference** | Auto-generated from schema | All YAML config keys with types, defaults, valid values, and descriptions. Generated from a schema definition (e.g., Pydantic model or JSON Schema) so docs and validation stay in sync. | All users |
| **Architecture overview** | Markdown + diagrams | Module dependency diagram, data flow diagram, extension point map. Aimed at developers who want to understand the system before contributing. | Developers |

### 16.2 Docstring Standard

All public functions and classes use Google-style docstrings:

```python
def binary_metrics(y_true: np.ndarray, prob1: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute binary classification metrics from predictions.

    Args:
        y_true: Ground truth labels (0 or 1), shape (N,).
        prob1: Predicted probability of class 1, shape (N,).
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        Dictionary with keys: tn, fp, fn, tp, sensitivity, specificity,
        ppv, npv, accuracy, auc, balanced_accuracy.

    Raises:
        ValueError: If y_true contains values other than 0 and 1.
    """
```

### 16.3 README Structure

The repository README serves as the entry point and must contain:

1. One-paragraph project description
2. Installation (pip, with dependency group options)
3. Quickstart (single command to run demo on synthetic data)
4. Link to full documentation
5. Citation (BibTeX entry once a paper is published)
6. License
7. Contributing link

### 16.4 Delivery

Documentation is part of Phase 7 deliverables but docstrings are written inline during all phases (Phases 0–6). The documentation site is built and deployed in Phase 7.

---

## 17. Out of Scope (Explicitly Excluded from MVP)

| Item | Reason |
|---|---|
| Federated learning | Significant infrastructure; future consideration for multi-center EANM studies |
| Real QPU execution | All quantum runs are simulator-only; hardware execution is a future extension |
| Web UI / dashboard | CLI + config + Jupyter notebooks are the interface; no web app |
| Automated hyperparameter optimization (AutoML) | Users set hyperparameters via config or use nested CV; no Optuna/Ray Tune integration in MVP |
| Image segmentation | The framework does supervised prediction, not segmentation (segmentation masks are inputs, not outputs) |
| External experiment tracking (MLflow, W&B) | File-based provenance is sufficient for MVP; integration hooks are a future extension |
| AI pipeline orchestration (n8n, Airflow) | Future plan to integrate the framework as a node/step in workflow automation platforms like n8n. The CLI + config-file architecture is designed to be pipeline-friendly (each stage is independently invocable with file-based I/O), but explicit n8n/Airflow adapters are not in MVP. |
| Subprocess-based dependency isolation | For MVP, incompatible dependency groups are handled via documentation and separate virtual environments. Automated subprocess orchestration is a future enhancement. |

---

## 18. Phased Delivery Plan

| Phase | Scope | Deliverables |
|---|---|---|
| **0** | Scaffolding | Project structure, pyproject.toml, registry, base classes, CLI skeleton, dependency compatibility matrix, prediction output contract implementation, synthetic test fixtures |
| **1** | I/O + Preprocessing | NIfTI loader, TIFF loader, tabular loader, pyradiomics wrapper, radiomics-tabular merger, unified Dataset object, splitting strategies. Unit tests for all loaders and encoders. |
| **2** | Classical Models | Port/rewrite tabular models (XGBoost, LightGBM, CatBoost, TabPFN, MLP), DEBI-NN subprocess wrapper, ensemble builder. Smoke tests per model. |
| **3** | Quantum Models | Port quantum kernel SVM, QCNN (muw/alt), add VQC, quantum encoding module, hybrid quantum-classical models. Smoke tests per model. |
| **4** | Imaging Models | 2D CNN (PyTorch), 3D CNN (MONAI/PyTorch), late fusion pipeline, imaging explainability (GradCAM). Smoke tests per model. |
| **5** | Explainability | SHAP/LIME wrappers (generalized), Integrated Gradients, quantum parameter-shift attribution, conformal prediction |
| **6** | Evaluation + Reporting | Metric registry, statistical tests, LaTeX report generator, publication figures, structured clinical reports |
| **7** | Testing + Documentation + Deployment | Full CI pipeline (GitHub Actions), integration tests, benchmark regression tests, Dockerfile, Singularity, API docs (Sphinx/MkDocs), user guide, model contributor guide, config reference, README |

---

## Appendix A: Source Codebase Inventory

### A.1 Benchmark Project (16 files)

`config.py`, `run_benchmark.py`, `download_benchmark_datasets.py`, `preprocess_benchmark.py`, `fold_generator.py`, `project_builder.py`, `settings_generator.py`, `debinn_runner.py`, `competitor_runner.py`, `ensemble_aggregator.py`, `results_collector.py`, `split_train_test.py`, `analysis.py`, `base_settings.csv`, `benchmark_log.txt`, `competitor_log.txt`

### A.2 EANM-AI-QC Project (25 Python files + 5 notebooks + 5 demo CSVs)

**Package**: `__init__.py` (root), `io/__init__.py`, `models/__init__.py`, `explain/__init__.py`
**CLI**: `qnm_qai.py`, `cli.py`
**Infrastructure**: `determinism.py`, `common.py`, `base.py`, `utils.py`
**I/O**: `tabular.py`, `nifti.py`, `encoding.py`
**Models**: `pl_kernel_svm.py`, `pl_qcnn_muw.py`, `pl_qcnn_alt.py`
**Explainability**: `shap_explain.py`, `lime_explain.py`
**Runner**: `runner.py`, `metrics.py`
**Examples**: `make_synthetic_nifti.py`, `build_tabular_from_fdb_ldb.py`, `run_explain_all.py`, `run_all_examples.sh`, `run_explain_all.sh`
**Notebooks**: `00_quickstart.ipynb` through `04_results_dashboard.ipynb`
**Demo data**: `FDB.csv`, `LDB.csv`, `real_train.csv`, `real_infer.csv`, `real_feature_map.csv`

---

## Appendix B: Open Decisions

| ID | Decision | Options | Status |
|---|---|---|---|
| D-1 | License | MIT, Apache-2.0, LGPL, BSD-3 | TBD |
| D-2 | Python version floor | 3.10, 3.11, 3.12 | Pending compatibility matrix |
| D-3 | Quantum backend(s) | PennyLane only vs. PennyLane + Qiskit | Start PennyLane; extend later |
| D-4 | MONAI integration depth | Required vs. optional fallback to PyTorch | Optional with fallback |
| D-5 | Plugin package format | namespace packages vs. entry_points | TBD in Phase 7 |
| D-6 | Repo name / package name | `eanm-ai-qc` (retain) vs. new name | **DECIDED**: `claryon` — CLARYON. EANM-AI-QC repo retained as educational hub. |
| D-7 | External experiment tracking | MLflow vs. W&B vs. file-only | File-only for MVP; hooks for external systems later |
| D-8 | Pipeline orchestration | n8n, Airflow, or none | Future; CLI designed to be pipeline-friendly |
| D-9 | Benchmark dataset scope | 28 existing datasets vs. add more (e.g., medical imaging benchmarks) | Start with 28; extend via config |
