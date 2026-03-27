# CLARYON

**The EANM AI Committee's git repository to support Nuclear Medicine research**

**CLassical-quantum AI for Reproducible Explainable OpeN-source medicine**

CLARYON is a YAML-driven machine learning framework that unifies classical, quantum, and deep learning models under a single reproducible pipeline. It supports tabular data and NIfTI medical images — with built-in preprocessing, explainability, statistical comparison, and publication-ready LaTeX reporting.

**Author**: Laszlo Papp, PhD — EANM AI Committee member, Applied Quantum Computing Group, Center for Medical Physics and Biomedical Engineering, Medical University of Vienna — laszlo.papp@meduniwien.ac.at

---

## Table of Contents

- [Installation](#installation)
- [Supported Data Types](#supported-data-types)
  - [Tabular Data](#tabular-data)
  - [NIfTI Medical Imaging](#nifti-medical-imaging)
  - [Data Fusion](#data-fusion)
- [Models](#models)
  - [Model–Data Compatibility](#modeldata-compatibility)
  - [Classical Models](#classical-models)
  - [Quantum Models](#quantum-models)
  - [Geometric Difference Score](#geometric-difference-score)
- [Quickstart: Tabular Workflow](#quickstart-tabular-workflow)
- [Quickstart: NIfTI Imaging Workflow](#quickstart-nifti-imaging-workflow)
- [Preprocessing](#preprocessing)
- [Quantum Best Practices](#quantum-best-practices)
- [Model Complexity Presets](#model-complexity-presets)
- [Running Benchmarks](#running-benchmarks)
- [Runtime Expectations](#runtime-expectations)
- [Explainability](#explainability)
- [Reporting](#reporting)
- [Inference on New Data](#inference-on-new-data)
- [Included Datasets](#included-datasets)
- [Configuration Reference](#configuration-reference)
- [Notebooks](#notebooks)
- [Project Structure](#project-structure)
- [Docker / Singularity](#docker--singularity)
- [Development](#development)
- [References](#references)
- [Citation](#citation)
- [License](#license)

---

## Installation

```bash
# Core (tabular models only)
pip install claryon

# With quantum models (PennyLane)
pip install claryon[quantum]

# With medical imaging (NIfTI, TIFF)
pip install claryon[imaging]

# With gradient boosting (XGBoost, LightGBM, CatBoost)
pip install claryon[boosting]

# With explainability (SHAP, LIME)
pip install claryon[explain]

# With deep learning (PyTorch CNNs)
pip install claryon[torch]

# Everything
pip install claryon[all]
```

**From source:**

```bash
git clone https://github.com/lpapp-muw/CLARYON.git
cd CLARYON
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

**Notes:**
- Python >= 3.10 required. Tested on 3.10, 3.11, 3.12.
- pyradiomics requires: `pip install numpy versioneer && pip install pyradiomics==3.0.1 --no-build-isolation`
- TabPFN downloads pretrained weights (~500 MB) on first use. Requires Python <= 3.11.
- PyTorch CPU is sufficient for quantum models. For large imaging, install with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

---

## Supported Data Types

CLARYON accepts two primary data types. Models are matched to data types via the `type` field in the config.

### Tabular Data

Semicolon-separated CSV with feature columns (`f0`, `f1`, ...) and a `label` column:

```
Key;f0;f1;f2;f3;label
S0000;5.1;3.5;1.4;0.2;0
S0001;7.0;3.2;4.7;1.4;1
```

Sources: radiomic features from pyradiomics, clinical parameters, lab values, any structured numeric data. The included PSMA-11 dataset (306 ADC + PSMA radiomic features from [68Ga]Ga-PSMA-11 PET/MR) is a representative example.

Config:
```yaml
data:
  tabular:
    path: datasets/wisconsin-breast-cancer/train.csv
    label_col: label
    sep: ";"
```

### NIfTI Medical Imaging

NIfTI volumes (.nii, .nii.gz) with optional binary masks. CLARYON pairs images and masks by filename pattern matching.

Typical use case: PET/CT or PET/MR volumes with lesion masks. The mask isolates the volume of interest (VOI) — only masked voxels are used.

Config:
```yaml
data:
  imaging:
    path: data/pet_volumes
    format: nifti
    image_pattern: "*PET*"
    mask_pattern: "*mask*"
```

**Three processing pathways** depending on model type:

| Model type | What happens to NIfTI data | Use case |
|---|---|---|
| `imaging` | Raw 3D tensors → PyTorch CNN | Classical deep learning on volumes |
| `tabular_quantum` | Volumes masked, flattened → amplitude encoding → quantum circuit | Quantum ML on imaging |
| `tabular` | Volumes masked, flattened → classical ML on voxel features | Classical ML on flattened volumes |

**Hard requirement**: All NIfTI volumes within a cohort must have the same dimensions (e.g., all 32×32×32). If volumes differ in size, CLARYON zero-pads smaller volumes to match the largest, and logs a warning. Consistent dimensions ensure spatial correspondence across patients.

**Qubit count**: For quantum models, amplitude encoding requires log2(n_voxels) qubits (rounded up to next power of 2). CLARYON logs the qubit count when loading NIfTI data. Example: 32×32×32 = 32768 voxels → 15 qubits.

### Data Fusion

Combine tabular features with imaging data. Both sources are loaded and concatenated (early fusion):

```yaml
data:
  tabular:
    path: data/radiomics.csv
    label_col: label
    sep: ";"
  imaging:
    path: data/pet_volumes
    format: nifti
    mask_pattern: "*mask*"
```

### Radiomics Extraction

CLARYON can extract radiomic features from NIfTI volumes using pyradiomics (IBSI-compliant):

```yaml
data:
  radiomics:
    extract: true
    config: configs/pyradiomics_default.yaml
```

---

## Models

CLARYON has 18 registered models plus an ensemble aggregator.

### Model–Data Compatibility

| Model | Type | Tabular | NIfTI (flattened) | NIfTI (3D volumes) |
|---|---|---|---|---|
| XGBoost | `tabular` | yes | yes | — |
| LightGBM | `tabular` | yes | yes | — |
| CatBoost | `tabular` | yes | yes | — |
| MLP | `tabular` | yes | yes | — |
| TabPFN | `tabular` | yes | yes | — |
| TabM | `tabular` | yes | yes | — |
| RealMLP | `tabular` | yes | yes | — |
| ModernNCA | `tabular` | yes | yes | — |
| CNN 2D | `imaging` | — | — | yes |
| CNN 3D | `imaging` | — | — | yes |
| Kernel SVM | `tabular_quantum` | yes | yes | — |
| Simplified Kernel SVM | `tabular_quantum` | yes | yes | — |
| QDC Hadamard | `tabular_quantum` | yes | yes | — |
| QDC SWAP | `tabular_quantum` | yes | yes | — |
| Quantum GP | `tabular_quantum` | yes | yes | — |
| QNN | `tabular_quantum` | yes | yes | — |
| QCNN-MUW | `tabular_quantum` | yes | yes | — |
| QCNN-ALT | `tabular_quantum` | yes | yes | — |
| Ensemble | `tabular` | yes | yes | — |

"NIfTI (flattened)" means volumes are flattened to feature vectors, then processed like tabular data — including amplitude encoding for quantum models.

### Classical Models

| Model | Backend | Key parameters |
|---|---|---|
| `xgboost` | XGBoost | n_estimators, max_depth, learning_rate |
| `lightgbm` | LightGBM | n_estimators, max_depth, learning_rate |
| `catboost` | CatBoost | iterations, depth, learning_rate |
| `mlp` | scikit-learn | hidden_layer_sizes, max_iter |
| `tabpfn` | TabPFN | (pretrained, no tuning) |
| `cnn_2d` | PyTorch | epochs, batch_size, n_conv_layers |
| `cnn_3d` | PyTorch | epochs, batch_size, n_conv_layers |
| `tabm` | PyTorch | epochs, lr |
| `realmlp` | PyTorch | epochs, lr |
| `modernnca` | PyTorch | epochs, lr |

### Quantum Models

All quantum models use PennyLane's `default.qubit` simulator. Data is amplitude-encoded: padded to 2^n and L2-normalized. Qubit count = log2(padded feature count).

| Model | Circuit design | Reference |
|---|---|---|
| `kernel_svm` | Amplitude embedding → Projector kernel → SVC | Havlicek et al., 2019 |
| `sq_kernel_svm` | Mottonen + adjoint Mottonen → Projector → linear prediction | Moradi et al., 2022 |
| `qdc_hadamard` | Ancilla + controlled Mottonen + Hadamard test → class-max similarity | Moradi et al., 2022 |
| `qdc_swap` | Two registers + CSWAP + ancilla → class-max similarity (uses 2n+1 qubits) | Moradi et al., 2022 |
| `quantum_gp` | Mottonen kernel → full GP posterior → sigmoid classification | Moradi et al., 2023 |
| `qnn` | Per-class Mottonen + Rot/CNOT layers → margin loss (PyTorch) | Moradi et al., 2023 |
| `qcnn_muw` | Amplitude embedding → conv/pool layers → ArbitraryUnitary → Projector | Moradi et al., under revision |
| `qcnn_alt` | Alternative conv/pool architecture → Projector | MedUni Wien design |

### Geometric Difference Score

The GDQ score (Huang et al., 2021) quantifies whether a quantum kernel provides a structurally different similarity measure from classical kernels. GDQ > 1.0 suggests potential quantum advantage.

```python
from claryon.evaluation.geometric_difference import quantum_advantage_analysis

analysis = quantum_advantage_analysis(K_Q, y_train, X_train)
print(analysis["recommendation"])   # classical_sufficient / quantum_advantage_likely / inconclusive
print(analysis["g_CQ"])             # geometric difference per classical kernel
```

Demonstrated in notebook `03_quantum_models.ipynb`.

---

## Quickstart: Tabular Workflow

```yaml
# configs/my_tabular.yaml
experiment:
  name: my_experiment
  seed: 42
  results_dir: Results/my_experiment
  complexity: medium

data:
  tabular:
    path: datasets/wisconsin-breast-cancer/train.csv
    label_col: label
    sep: ";"

preprocessing:
  zscore: true                # auto-skipped for quantum models
  feature_selection: true
  spearman_threshold: 0.8
  max_features: 8             # recommended for quantum (3 qubits)

cv:
  strategy: kfold
  n_folds: 5
  seeds: [42, 123]

models:
  - name: xgboost
    type: tabular
  - name: lightgbm
    type: tabular
  - name: kernel_svm
    type: tabular_quantum
  - name: qcnn_muw
    type: tabular_quantum

explainability:
  shap: true
  lime: true

evaluation:
  metrics: [bacc, auc, sensitivity, specificity]
  statistical_tests: [friedman]

reporting:
  markdown: true
  latex: true
  figures: true
```

```bash
claryon -v run -c configs/my_tabular.yaml
```

Results:
```
Results/my_experiment/
├── metrics_summary.csv       # model;bacc;bacc_std;auc;auc_std;...
├── report.md
├── methods.tex               # structured prose for publication
├── results.tex               # LaTeX table with mean ± std
├── references_needed.txt     # BibTeX keys used
├── run_info.json
├── config_used.yaml
├── xgboost/seed_42/fold_0/
│   ├── Predictions.csv
│   └── preprocessing_state.json
└── kernel_svm/seed_42/fold_0/
    ├── Predictions.csv
    └── preprocessing_state.json
```

---

## Quickstart: NIfTI Imaging Workflow

### Classical CNN on PET volumes

```yaml
experiment:
  name: pet_cnn
  seed: 42
  complexity: medium

data:
  imaging:
    path: datasets/nifti_demo
    format: nifti
    image_pattern: "*PET*"
    mask_pattern: "*mask*"

cv:
  strategy: kfold
  n_folds: 3
  seeds: [42]

models:
  - name: cnn_3d
    type: imaging
    params:
      epochs: 50
      batch_size: 4

evaluation:
  metrics: [bacc, auc]
```

### Quantum QCNN on PET VOIs

The core nuclear medicine quantum workflow (Moradi et al., 2022, 2023; under revision). NIfTI volumes are loaded, masked (voxels outside the mask are set to zero), flattened to a feature vector, then amplitude-encoded for quantum circuits.

```yaml
experiment:
  name: pet_quantum
  seed: 42
  complexity: medium

data:
  imaging:
    path: data/pet_voi
    format: nifti
    mask_pattern: "*mask*"

preprocessing:
  feature_selection: true
  max_features: 8             # optional: reduces qubit count for tabular radiomics

cv:
  strategy: kfold
  n_folds: 5
  seeds: [42, 123]

models:
  - name: kernel_svm
    type: tabular_quantum           # flattened VOI → amplitude encoding
  - name: qcnn_muw
    type: tabular_quantum
  - name: xgboost
    type: tabular                   # classical comparison on same features
```

**Hard requirement**: All NIfTI volumes in a cohort must have the same dimensions. The mask can vary per patient (different organ/lesion shapes), but the volume grid must be identical (e.g., all 32×32×32). CLARYON multiplies each volume by its mask (zeros outside the mask) and flattens the full grid.

**Qubit count** is determined by the total voxel count of the volume (not the mask), rounded up to the next power of 2:

| VOI dimensions | Total voxels | Qubits | Notes |
|---|---|---|---|
| 4×4×4 | 64 | 6 | Fast, ideal for testing |
| 8×8×8 | 512 | 9 | Moderate runtime |
| 16×16×16 | 4096 | 12 | Hours per fold |
| 32×32×32 | 32768 | 15 | Feasible with high compute (used in Moradi et al., under revision) |
| 64×64×64 | 262144 | 18 | Very demanding, cluster recommended |

---

## Preprocessing

All preprocessing runs inside the cross-validation loop, fitting on training data only. State is saved per fold for reproducible inference.

### Z-Score Normalization

Features standardized to zero mean, unit variance. **Automatically skipped for quantum models** — amplitude encoding L2-normalizes the feature vector, and prior z-score distorts quantum kernel geometry by 30-40%.

### mRMR Feature Selection

Features with Spearman correlation above threshold are clustered as redundant. Most label-correlated feature per cluster is kept.

```yaml
preprocessing:
  feature_selection: true
  spearman_threshold: 0.8
  max_features: 8            # optional hard cap (critical for quantum)
```

**Warning**: mRMR only removes correlated features. If features are uncorrelated, mRMR removes nothing. Always set `max_features` for quantum models.

### Binary Grouping

```yaml
binary_grouping:
  enabled: true
  positive: [3, 4]            # e.g., ISUP grades 3+4 → high risk
  negative: [1, 2]
```

### Image Normalization

```yaml
preprocessing:
  image_normalization: per_image       # each volume to [0, 1]
  # OR: cohort_global                  # global min/max from training set
```

---

## Quantum Best Practices

### Feature count → qubit count

| Features after mRMR | Padded to | Qubits | Zero-pad waste | Recommendation |
|---|---|---|---|---|
| 4 | 4 | 2 | 0% | Ideal |
| 5–8 | 8 | 3 | 0–37% | **Recommended sweet spot** |
| 9–16 | 16 | 4 | 0–44% | Acceptable |
| 17–32 | 32 | 5 | 0–47% | Slow on simulator |
| 33–64 | 64 | 6+ | up to 48% | Not recommended on single CPU |

**For tabular data**, control qubit count with `max_features`:

```yaml
preprocessing:
  max_features: 8            # 3 qubits — best for simulator
```

**For NIfTI imaging**, qubit count is determined by the volume dimensions (total voxel count). The user controls this by choosing an appropriate VOI size during data preparation. mRMR feature selection can optionally further reduce the feature count.

### Why fewer qubits often improves results

Amplitude encoding maps features to the unit hypersphere in 2^n dimensions. Zero-padded features dilute the signal. Observed on real medical data:

| Dataset | Features → Qubits | Zero-pad waste | Quantum BACC |
|---|---|---|---|
| Iris | 4 → 2 | 0% | 1.00 |
| Wisconsin | 13 → 4 | 19% | 0.76 |
| HCC (no max_features) | 49 → 6 | 23% | 0.52 |
| PSMA (no max_features) | 40 → 6 | 37% | 0.50 |

### Model-specific notes

| Model | Note |
|---|---|
| `sq_kernel_svm` | Returns BACC ~0.500 on balanced datasets — known limitation. Use `kernel_svm` instead. |
| `qdc_swap` | Uses 2n+1 qubits. At 5+ data qubits, infeasible on simulator. Exclude from large-feature configs. |
| `qcnn_muw`, `qcnn_alt` | Need ≥100 epochs. Use `complexity: medium` minimum. |
| `quantum_gp` | Most robust quantum model on real data. |

---

## Model Complexity Presets

```yaml
experiment:
  complexity: medium          # recommended for most studies
```

| Preset | Quantum epochs | Quantum LR | Classical estimators | Use case |
|---|---|---|---|---|
| `quick` | 5 | 0.05 | 50 | CI/testing only |
| `small` | 30 | 0.02 | 200 | Fast exploratory |
| `medium` | 100 | 0.01 | 500 | **Recommended for publication** |
| `large` | 300 | 0.005 | 1000 | Final results, small datasets |
| `exhaustive` | 500 | 0.002 | 2000 | Cluster only |
| `auto` | varies | varies | varies | Estimates per model (approximate) |

Per-model override:
```yaml
models:
  - name: qcnn_muw
    preset: large
  - name: xgboost
    params:
      n_estimators: 1000
```

**Auto mode note**: Runtime estimates for quantum models may underestimate significantly. Use explicit presets for predictable runtimes.

---

## Running Benchmarks

```bash
# All 3 datasets
bash scripts/run_benchmark.sh

# Single dataset
bash scripts/run_benchmark.sh wisconsin

# In a screen session (recommended)
screen -S benchmark
bash scripts/run_benchmark.sh
# Detach: Ctrl+A then D | Reattach: screen -r benchmark
```

Available: `wisconsin`, `hcc`, `psma11` (3 of the 6 included datasets). Uses `max_features: 8` (3 qubits). Iris is excluded (trivial smoke test), cervical cancer is excluded (858 samples at 5 qubits takes days), and NIfTI demo is excluded (synthetic CNN-only data). Cervical can be run classical-only via `configs/eanm_abstract/cervical.yaml`.

---

## Runtime Expectations

### Per-fold (observed, `complexity: medium`, single CPU)

| Model | ~150 samples, 4q | ~570 samples, 4q | ~860 samples, 5q |
|---|---|---|---|
| Classical (XGB/LGB/CB/MLP) | < 1 sec | < 1 sec | 1–2 sec |
| kernel_svm | < 1 min | 3 min | 15 min |
| sq_kernel_svm | 1 min | 11 min | 45 min |
| qdc_hadamard | 1 min | 10 min | 80 min |
| qdc_swap | 2 min | 20 min | **infeasible** (11q) |
| quantum_gp | 1 min | 10 min | 40 min |
| qnn | 3 min | 20 min | 1–2 hr |
| qcnn_muw / qcnn_alt | 5 min | 30 min | 2–4 hr |
| cnn_3d | 2 min | 10 min | 30 min (GPU) |

Multiply by folds × seeds (e.g., 5 × 3 = 15).

### Total experiment (observed, 5-fold CV × 3 seeds)

| Dataset | Samples | Qubits | Classical | Quantum | Total |
|---|---|---|---|---|---|
| Iris | 150 | 2 | < 1 min | 30 min | ~30 min |
| Wisconsin | 569 | 3 | < 1 min | 2–4 hr | ~4 hr |
| HCC Survival | 165 | 3 | < 1 min | 1–3 hr | ~3 hr |
| PSMA-11 | 133 | 3 | < 1 min | 1–3 hr | ~3 hr |
| Cervical | 858 | 5 | < 1 min | **3–5 days** | **3–5 days** |

Use `max_features: 8` (3 qubits) for all quantum experiments. Exclude `qdc_swap` when data qubits > 4.

---

## Explainability

SHAP and LIME work on all models, including quantum.

```yaml
explainability:
  shap: true
  lime: true
  max_features: 32
  max_test_samples: 5
```

Outputs: `shap_bar.png`, `shap_beeswarm.png`, `shap_waterfall_sample_*.png`, `lime_explanation_sample_*.png`. GradCAM available for CNN models.

---

## Reporting

Auto-generated structured methods and results for publication:

```yaml
reporting:
  markdown: true
  latex: true
  figures: true
  figure_dpi: 300
```

Outputs: `methods.tex` (prose), `results.tex` (table), `references_needed.txt` (BibTeX keys), `report.md`.

---

## Inference on New Data

```bash
claryon infer \
    --model-dir Results/my_experiment/xgboost/seed_42/fold_0/ \
    --input data/new_patients.csv \
    --output predictions.csv
```

Loads saved model + preprocessing state and applies identically to training.

---

## Included Datasets

All pre-processed, semicolon-separated, ready to use.

| Dataset | Path | Samples | Features | Domain |
|---|---|---|---|---|
| Iris | `datasets/iris/` | 150 | 4 | Demo / smoke test |
| Wisconsin Breast Cancer | `datasets/wisconsin-breast-cancer/` | 569 | 30 | Oncology (UCI, CC BY 4.0) |
| Cervical Cancer | `datasets/cervical-cancer/` | 858 | 26 | Oncology (UCI, CC BY 4.0) |
| HCC Survival | `datasets/hcc-survival/` | 165 | 49 | Oncology (Kaggle, CC BY-NC-SA 4.0) |
| PSMA-11 Radiomics | `datasets/psma11/` | 133 | 306 | Nuclear medicine (OSF, open) |
| NIfTI Demo | `datasets/nifti_demo/` | 32 | volumetric | Synthetic (pipeline validation) |

See `datasets/DATA_SOURCES.md` for full attribution and licenses.

---

## Configuration Reference

Full reference: [docs/config_reference.md](docs/config_reference.md). Key sections:

| Section | Key parameters |
|---|---|
| `experiment` | `name`, `seed`, `results_dir`, `complexity`, `max_runtime_minutes` |
| `data.tabular` | `path`, `label_col`, `sep`, `id_col` |
| `data.imaging` | `path`, `format`, `image_pattern`, `mask_pattern` |
| `preprocessing` | `zscore`, `feature_selection`, `spearman_threshold`, `max_features` |
| `cv` | `strategy`, `n_folds`, `seeds`, `test_size` |
| `models[]` | `name`, `type`, `preset`, `params`, `enabled` |
| `explainability` | `shap`, `lime`, `grad_cam`, `max_features`, `max_test_samples` |
| `evaluation` | `metrics`, `statistical_tests`, `confidence_level` |
| `reporting` | `markdown`, `latex`, `figures`, `figure_dpi` |

Available models: `xgboost`, `lightgbm`, `catboost`, `mlp`, `tabpfn`, `cnn_2d`, `cnn_3d`, `kernel_svm`, `sq_kernel_svm`, `qdc_hadamard`, `qdc_swap`, `quantum_gp`, `qnn`, `qcnn_muw`, `qcnn_alt`, `tabm`, `realmlp`, `modernnca`.

---

## Notebooks

| Notebook | Content |
|---|---|
| `01_quickstart.ipynb` | Install, load iris, run XGBoost, inspect predictions |
| `02_tabular_classification.ipynb` | Full tabular workflow with multiple models |
| `03_quantum_models.ipynb` | All quantum models on iris, GDQ score demo |
| `04_nifti_imaging.ipynb` | NIfTI + masks, 3D CNN |
| `05_explainability.ipynb` | SHAP + LIME on classical and quantum models |
| `06_results_dashboard.ipynb` | Metrics visualization and statistical comparison |
| `07_radiomics.ipynb` | Radiomics extraction, merge, train |
| `08_custom_model_guide.ipynb` | How to add a new model using @register |

---

## Project Structure

```
claryon/
├── cli.py                    # CLI (run, infer, list-models, validate-config)
├── config_schema.py          # Pydantic YAML config validation
├── pipeline.py               # 8-stage orchestrator
├── registry.py               # @register decorator
├── determinism.py            # Seed + thread control
├── progress.py               # CLI progress display
├── safety.py                 # Resource estimation + OOM protection
├── inference.py              # Inference on new data
├── io/                       # Data loaders (tabular, NIfTI, TIFF, predictions)
├── preprocessing/            # Z-score, mRMR, image normalization, state persistence
├── encoding/                 # Amplitude + angle quantum encoding
├── models/
│   ├── classical/            # XGBoost, LightGBM, CatBoost, MLP, CNN
│   ├── quantum/              # Kernel SVM, QDC, GP, QNN, QCNN
│   ├── presets.yaml          # Complexity preset definitions
│   └── ensemble.py           # Softmax averaging
├── explainability/           # SHAP, LIME, GradCAM, plots
├── evaluation/               # Metrics, Friedman/Nemenyi, GDQ, figures
└── reporting/                # Structured LaTeX, Markdown, BibTeX

scripts/
└── run_benchmark.sh          # Benchmark runner for included datasets

datasets/                     # Pre-processed, ready-to-use datasets
configs/                      # Example + benchmark YAML configs
docs/                         # Architecture, config reference, model guide, user guide
examples/notebooks/           # 8 tutorial Jupyter notebooks
tests/                        # Unit + integration tests
```

---

## Adding a New Model

1. Create `claryon/models/classical/mymodel_.py` (or `quantum/`)
2. Subclass `ModelBuilder`, implement `fit()`, `predict_proba()`
3. Decorate with `@register("model", "mymodel")`
4. Auto-discovered — no other code changes needed
5. Optionally add prose to `claryon/reporting/method_descriptions.yaml`

See notebook `08_custom_model_guide.ipynb`.

---

## Docker / Singularity

```bash
# Docker (CPU)
docker build -t claryon .
docker run -v $(pwd)/data:/app/data -v $(pwd)/Results:/app/Results \
    claryon run -c configs/my_config.yaml

# Docker (GPU)
docker build -f Dockerfile.gpu -t claryon-gpu .
docker run --gpus all -v ... claryon-gpu run -c configs/my_config.yaml

# Singularity (HPC)
singularity build claryon.sif singularity.def
singularity run claryon.sif run -c configs/my_config.yaml
```

---

## Development

```bash
python -m pytest tests/ -q --timeout=300
ruff check claryon/ tests/
```

CI runs on Python 3.10–3.12 via GitHub Actions.

---

## References

- Papp L, Visvikis D, Sollini M, Shi K, Kirienko M. "The Dawn of Quantum AI in Nuclear Medicine: an EANM Perspective." *The EANM Journal*, 2026 (in revision). CLARYON is the official code repository for this manuscript (via [EANM-AI-QC](https://github.com/lpapp-muw/EANM-AI-QC)).
- Moradi S, Brandner C, Spielvogel C, Krajnc D, Hillmich S, Wille R, Drexler W, Papp L. "Clinical data classification with noisy intermediate scale quantum computers." *Scientific Reports* 12, 1851 (2022). https://doi.org/10.1038/s41598-022-05971-9
- Moradi S, Spielvogel C, Krajnc D, Brandner C, Hillmich S, Wille R, Traub-Weidinger T, Li X, Hacker M, Drexler W, Papp L. "Error mitigation enables PET radiomic cancer characterization on quantum computers." *Eur J Nucl Med Mol Imaging* 50, 3826-3837 (2023). https://doi.org/10.1007/s00259-023-06362-6
- Moradi S, et al. "Quantum Convolutional Neural Networks for Predicting ISUP Grade risk in [68Ga]Ga-PSMA Primary Prostate Cancer Patients." Under revision.
- Huang H-Y, Broughton M, Mohseni M, Babbush R, Boixo S, Neven H, McClean JR. "Power of data in quantum machine learning." *Nature Communications* 12, 2631 (2021). https://doi.org/10.1038/s41467-021-22539-9
- Papp L, Spielvogel CP, Grubmuller B, et al. "Supervised machine learning enables non-invasive lesion characterization in primary prostate cancer with [68Ga]Ga-PSMA-11 PET/MRI." *Eur J Nucl Med Mol Imaging* 48, 1795-1805 (2021). https://doi.org/10.1007/s00259-020-05140-y

---

## Citation

```bibtex
@article{papp2026dawn,
  author       = {Papp, Laszlo and Visvikis, Dimitris and Sollini, Martina
                  and Shi, Kuangyu and Kirienko, Margarita},
  title        = {The Dawn of Quantum {AI} in Nuclear Medicine: an {EANM}
                  Perspective},
  journal      = {The EANM Journal},
  year         = {2026},
  note         = {In revision}
}

@software{claryon2026,
  author       = {Papp, Laszlo},
  title        = {{CLARYON}: Classical-quantum {AI} for Reproducible
                  Explainable Open-source Medicine},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/lpapp-muw/CLARYON}
}
```

---

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Medical University of Vienna (MedUni Wien), EANM AI Committee.
