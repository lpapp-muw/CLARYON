# CLARYON

**CLassical-quantum AI for Reproducible Explainable OpeN-source medicine**

CLARYON is a YAML-driven machine learning framework that unifies classical, quantum, and deep learning models under a single reproducible pipeline. It supports tabular data and NIfTI medical images — with built-in preprocessing, explainability, statistical comparison, and publication-ready LaTeX reporting.

**Author**: Laszlo Papp, PhD — EANM AI Committee member, Applied Quantum Computing Group, Center for Medical Physics and Biomedical Engineering, Medical University of Vienna — laszlo.papp@meduniwien.ac.at

---

## Table of Contents

- [Installation](#installation)
- [Supported Data Types](#supported-data-types)
- [Models](#models)
- [Quantum Encoding Strategies](#quantum-encoding-strategies)
- [Quickstart: Tabular Workflow](#quickstart-tabular-workflow)
- [Quickstart: NIfTI Imaging Workflow](#quickstart-nifti-imaging-workflow)
- [Preprocessing](#preprocessing)
- [Quantum Best Practices](#quantum-best-practices)
- [Model Complexity Presets](#model-complexity-presets)
- [Running Benchmarks](#running-benchmarks)
- [Benchmark Results](#benchmark-results)
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

Config:
```yaml
data:
  imaging:
    path: data/pet_volumes              # your NIfTI directory
    format: nifti
    image_pattern: "*PET*"
    mask_pattern: "*mask*"
```

**Three processing pathways** depending on model type:

| Model type | What happens to NIfTI data | Use case |
|---|---|---|
| `imaging` | Raw 3D tensors to PyTorch CNN | Classical deep learning on volumes |
| `tabular_quantum` | Volumes masked, flattened, amplitude-encoded | Quantum ML on imaging |
| `tabular` | Volumes masked, flattened, classical ML on voxel features | Classical ML on flattened volumes |

**Hard requirement**: All NIfTI volumes within a cohort must have the same dimensions (e.g., all 32x32x32). If volumes differ in size, CLARYON zero-pads smaller volumes to match the largest, and logs a warning.

**Qubit count**: For quantum models, amplitude encoding requires log2(n_voxels) qubits (rounded up to next power of 2). Example: 32x32x32 = 32768 voxels = 15 qubits.

### Radiomics Extraction

CLARYON can extract radiomic features from NIfTI volumes using pyradiomics (IBSI-compliant):

```yaml
data:
  radiomics:
    extract: true
    config: my_pyradiomics_config.yaml    # user-provided pyradiomics settings
```

---

## Models

CLARYON has 16 registered models plus an ensemble aggregator.

### Model-Data Compatibility

| Model | Type | Tabular | NIfTI (flattened) | NIfTI (3D volumes) |
|---|---|---|---|---|
| XGBoost | `tabular` | yes | yes | - |
| LightGBM | `tabular` | yes | yes | - |
| CatBoost | `tabular` | yes | yes | - |
| MLP | `tabular` | yes | yes | - |
| TabPFN | `tabular` | yes | yes | - |
| TabM | `tabular` | yes | yes | - |
| RealMLP | `tabular` | yes | yes | - |
| ModernNCA | `tabular` | yes | yes | - |
| CNN 2D | `imaging` | - | - | yes |
| CNN 3D | `imaging` | - | - | yes |
| **Angle PQK SVM** | **`tabular`** | **yes** | - | - |
| Projected Kernel SVM | `tabular_quantum` | yes | yes | - |
| Kernel SVM | `tabular_quantum` | yes | yes | - |
| QDC Hadamard | `tabular_quantum` | yes | yes | - |
| Quantum GP | `tabular_quantum` | yes | yes | - |
| QNN | `tabular_quantum` | yes | yes | - |
| QCNN-MUW | `tabular_quantum` | yes | yes | - |
| QCNN-ALT | `tabular_quantum` | yes | yes | - |
| Ensemble | `tabular` | yes | yes | - |

Note: `angle_pqk_svm` uses `type: tabular` because it handles quantum encoding internally (angle encoding, not amplitude). It receives z-scored features from the pipeline.

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

CLARYON provides two families of quantum models, distinguished by their encoding strategy:

**Angle-encoded models** (recommended for tabular radiomic data):

| Model | Circuit design | Encoding | Reference |
|---|---|---|---|
| `angle_pqk_svm` | AngleEmbedding, Pauli measurement, RBF kernel, SVC | Angle (1 qubit/feature) | Huang et al., 2021; Shaydulin and Wild, 2022 |

**Amplitude-encoded models** (for NIfTI imaging, and for comparative evaluation on tabular data):

| Model | Circuit design | Encoding | Reference |
|---|---|---|---|
| `projected_kernel_svm` | AmplitudeEmbedding, Pauli measurement, RBF, SVC | Amplitude | Huang et al., 2021 |
| `kernel_svm` | AmplitudeEmbedding, Projector (fidelity) kernel, SVC | Amplitude | Havlicek et al., 2019 |
| `qdc_hadamard` | Ancilla + controlled Mottonen + Hadamard test | Amplitude | Moradi et al., 2022 |
| `quantum_gp` | Mottonen kernel, full GP posterior, sigmoid classification | Amplitude | Moradi et al., 2023 |
| `qnn` | Per-class Mottonen + Rot/CNOT layers, margin loss | Amplitude | Moradi et al., 2023 |
| `qcnn_muw` | Amplitude embedding, conv/pool layers, ArbitraryUnitary | Amplitude | Moradi et al., under revision |
| `qcnn_alt` | Alternative conv/pool architecture | Amplitude | MedUni Wien design |

For tabular radiomic datasets, the angle-encoded `angle_pqk_svm` substantially outperforms amplitude-encoded quantum models (see [Benchmark Results](#benchmark-results)). Amplitude-encoded models remain available for comparative evaluation and for NIfTI imaging workflows where amplitude encoding is the only viable option (angle encoding would require one qubit per voxel, which is infeasible for typical VOI sizes).

### Geometric Difference Score

The GDQ score (Huang et al., 2021) quantifies whether a quantum kernel provides a structurally different similarity measure from classical kernels.

```python
from claryon.evaluation.geometric_difference import quantum_advantage_analysis

analysis = quantum_advantage_analysis(K_Q, y_train, X_train)
print(analysis["recommendation"])   # classical_sufficient / quantum_advantage_likely / inconclusive
print(analysis["g_CQ"])             # geometric difference per classical kernel
```

Note: GDQ evaluates fidelity (amplitude-encoded) kernels only. It does not apply to projected quantum kernels or training-based models (QNN, QCNN).

---

## Quantum Encoding Strategies

The choice of quantum encoding has a larger impact on performance than the choice of quantum model.

### Angle Encoding (recommended for tabular data)

Each feature is mapped to a dedicated qubit via RY(bandwidth * x_i). No L2 normalization, no information loss. Z-score standardization is applied before encoding (unlike amplitude encoding).

- Qubits needed: 1 per feature (e.g., 8 features = 8 qubits)
- Key hyperparameter: `bandwidth` (controls kernel sensitivity; default 0.5)
- Used by: `angle_pqk_svm`

### Amplitude Encoding (required for NIfTI, available for tabular)

The feature vector is zero-padded to the next power of 2 and L2-normalized into a quantum state. Z-score is NOT applied (it degrades kernel geometry).

- Qubits needed: log2(features) (e.g., 8 features = 3 qubits)
- Used by: all `tabular_quantum` models
- Required for NIfTI (angle encoding would need 32768 qubits for a 32x32x32 VOI)

### Why angle encoding outperforms on tabular data

Amplitude encoding's L2 normalization forces all data onto the unit hypersphere, destroying per-feature magnitude information. This causes quantum kernels to concentrate: all pairwise similarities become nearly identical. On our radiomic datasets, amplitude-encoded models achieved BACC 0.63-0.76, while the same kernel measurement with angle encoding achieved BACC 0.68-0.96 (see [Benchmark Results](#benchmark-results)).

---

## Quickstart: Tabular Workflow

```yaml
# configs/example_tabular.yaml (shipped with CLARYON)
experiment:
  name: tabular_binary_example
  seed: 42
  results_dir: Results/tabular_binary
  complexity: medium

data:
  tabular:
    path: datasets/wisconsin-breast-cancer/train.csv
    label_col: label
    sep: ";"

preprocessing:
  zscore: true
  feature_selection: true
  spearman_threshold: 0.8
  max_features: 8

cv:
  strategy: kfold
  n_folds: 5
  seeds: [42, 123]

models:
  - name: xgboost
    type: tabular
  - name: lightgbm
    type: tabular
  - name: catboost
    type: tabular
  - name: angle_pqk_svm
    type: tabular

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
claryon -v run -c configs/example_tabular.yaml
```

Results:
```
Results/tabular_binary/
├── metrics_summary.csv
├── report.md
├── methods.tex
├── results.tex
├── references_needed.txt
├── run_info.json
├── config_used.yaml
├── xgboost/seed_42/fold_0/
│   ├── Predictions.csv
│   └── preprocessing_state.json
└── angle_pqk_svm/seed_42/fold_0/
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

NIfTI volumes are loaded, masked, flattened, then amplitude-encoded for quantum circuits. Note: NIfTI quantum models use amplitude encoding (the only viable option for volumetric data).

```yaml
experiment:
  name: pet_quantum
  seed: 42
  complexity: medium

data:
  imaging:
    path: data/pet_voi                    # your NIfTI VOI directory
    format: nifti
    mask_pattern: "*mask*"

cv:
  strategy: kfold
  n_folds: 5
  seeds: [42, 123]

models:
  - name: kernel_svm
    type: tabular_quantum
  - name: qcnn_muw
    type: tabular_quantum
  - name: xgboost
    type: tabular
```

**Qubit count** is determined by the total voxel count:

| VOI dimensions | Total voxels | Qubits | Notes |
|---|---|---|---|
| 4x4x4 | 64 | 6 | Fast, ideal for testing |
| 8x8x8 | 512 | 9 | Moderate runtime |
| 16x16x16 | 4096 | 12 | Hours per fold |
| 32x32x32 | 32768 | 15 | Feasible with high compute (Moradi et al., under revision) |
| 64x64x64 | 262144 | 18 | Very demanding, cluster recommended |

---

## Preprocessing

All preprocessing runs inside the cross-validation loop, fitting on training data only. State is saved per fold for reproducible inference.

### Z-Score Normalization

Features standardized to zero mean, unit variance.

- **Automatically skipped for amplitude-encoded quantum models** (`type: tabular_quantum`). Amplitude encoding L2-normalizes the feature vector, and prior z-score distorts quantum kernel geometry by 30-40%.
- **Applied normally for angle-encoded quantum models** (`angle_pqk_svm` uses `type: tabular`). Angle encoding benefits from normalized feature scales.

### mRMR Feature Selection

Features with Spearman correlation above threshold are clustered as redundant. Most label-correlated feature per cluster is kept.

```yaml
preprocessing:
  feature_selection: true
  spearman_threshold: 0.8
  max_features: 8
```

**Warning**: mRMR only removes correlated features. If features are uncorrelated, mRMR removes nothing. Always set `max_features` for quantum models.

### Binary Grouping

```yaml
binary_grouping:
  enabled: true
  positive: [3, 4]
  negative: [1, 2]
```

### Image Normalization

```yaml
preprocessing:
  image_normalization: per_image
```

---

## Quantum Best Practices

### Angle-encoded models (tabular data)

Use `angle_pqk_svm` with `max_features: 8` (8 qubits). This is the recommended quantum model for tabular radiomic datasets.

```yaml
preprocessing:
  max_features: 8
models:
  - name: angle_pqk_svm
    type: tabular
```

The `bandwidth` parameter (default 0.5) controls kernel sensitivity. Lower values preserve finer distinctions between samples. A sweep across {0.5, 1.0, 2.0, 3.0, 5.0} showed optimal performance at bandwidth=0.5 on all three medical datasets, with monotonic degradation at higher values.

### Amplitude-encoded models (NIfTI imaging, comparative tabular)

For NIfTI data, amplitude encoding is the only option. For tabular data, amplitude-encoded models are available for comparative evaluation and for reproducing published results (Moradi et al., 2022, 2023).

| Features after mRMR | Padded to | Qubits (amplitude) | Qubits (angle) | Recommendation |
|---|---|---|---|---|
| 4 | 4 | 2 | 4 | Either encoding works |
| 5-8 | 8 | 3 | 5-8 | **Recommended sweet spot** |
| 9-16 | 16 | 4 | 9-16 | Acceptable |
| 17-32 | 32 | 5 | 17-32 | Slow on simulator |
| 33-64 | 64 | 6+ | 33-64 | Not recommended on single CPU |

### Model-specific notes

| Model | Note |
|---|---|
| `angle_pqk_svm` | Recommended for tabular data. O(N) circuit evaluations. ~5-10 seconds per fold. |
| `qcnn_muw`, `qcnn_alt` | Need >=100 epochs. Use `complexity: medium` minimum. |
| `quantum_gp` | Most robust amplitude-encoded model. Provides uncertainty estimates. |
| `kernel_svm` | O(N^2) fidelity kernel. Slow beyond ~500 samples. |

---

## Model Complexity Presets

```yaml
experiment:
  complexity: medium
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

---

## Running Benchmarks

The primary entry point for benchmarking on the included medical datasets:

```bash
# All 3 medical datasets (Wisconsin, HCC, PSMA-11)
bash scripts/run_benchmark.sh

# Single dataset
bash scripts/run_benchmark.sh wisconsin

# In a screen session (recommended)
screen -S benchmark
bash scripts/run_benchmark.sh
# Detach: Ctrl+A then D | Reattach: screen -r benchmark
```

This runs all classical and amplitude-encoded quantum models with `max_features: 8`, `complexity: medium`, 5-fold CV x 3 seeds. Configs: `configs/eanm_abstract/<dataset>_q8.yaml`.

To benchmark the angle-encoded quantum model:

```bash
claryon -v run -c configs/eanm_abstract/angle_pqk_wisconsin_q8.yaml
claryon -v run -c configs/eanm_abstract/angle_pqk_hcc_q8.yaml
claryon -v run -c configs/eanm_abstract/angle_pqk_psma11_q8.yaml
```

For NIfTI CNN benchmarking:

```bash
claryon -v run -c configs/nifti_cnn.yaml
```

---

## Benchmark Results

All results: 5-fold stratified CV x 3 seeds = 15 folds. `max_features: 8`, `complexity: medium`.

### Angle-encoded PQK SVM (recommended quantum model)

| Dataset | Samples | Best Classical (BACC) | angle_pqk_svm (BACC) | Gap | AUC |
|---|---|---|---|---|---|
| **Wisconsin** | 569 | 0.963 (MLP) | **0.964** | **+0.001** | 0.995 |
| **HCC Survival** | 165 | 0.688 (CatBoost) | **0.692** | **+0.004** | 0.773 |
| **PSMA-11** | 133 | 0.771 (CatBoost) | 0.756 | -0.015 | 0.823 |
| **Iris** | 150 | 0.995 (XGBoost) | **1.000** | +0.005 | 1.000 |

Bandwidth = 0.5 (optimal; monotonic degradation at higher values). Runtime: ~5-10 seconds per fold.

### Amplitude-encoded models (max_features=8, 3 qubits)

For comparative evaluation. On tabular radiomic datasets, amplitude encoding may underperform due to L2 normalization destroying per-feature magnitude information.

| Dataset | Best Classical (BACC) | Best Amplitude Quantum (BACC) | Model | Gap |
|---|---|---|---|---|
| Wisconsin (569) | 0.963 (MLP) | 0.757 (quantum_gp) | quantum_gp | -0.206 |
| HCC (165) | 0.688 (CatBoost) | 0.632 (quantum_gp) | quantum_gp | -0.056 |
| PSMA-11 (133) | 0.771 (CatBoost) | 0.634 (qdc_hadamard) | qdc_hadamard | -0.137 |

### Key finding

Angle encoding with projected quantum kernels closes approximately 90% of the quantum-classical performance gap on tabular radiomic data compared to amplitude-encoded quantum models. The improvement comes entirely from the encoding strategy: angle encoding preserves per-feature magnitude information that amplitude encoding destroys via L2 normalization.

---

## Runtime Expectations

### Per-fold (observed, `complexity: medium`, single CPU, max_features=8)

| Model | ~150 samples | ~570 samples |
|---|---|---|
| Classical (XGB/LGB/CB/MLP) | < 1 sec | < 1 sec |
| **angle_pqk_svm** (8 qubits) | **< 5 sec** | **5-10 sec** |
| kernel_svm (3 qubits) | < 1 min | 3 min |
| qdc_hadamard (3 qubits) | 1 min | 10 min |
| quantum_gp (3 qubits) | 1 min | 10 min |
| qnn (3 qubits) | 3 min | 20 min |
| qcnn_muw / qcnn_alt (3 qubits) | 5 min | 30 min |
| cnn_3d | 2 min | 10 min (GPU) |

Multiply by folds x seeds (e.g., 5 x 3 = 15).

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

For projected quantum kernel models (`angle_pqk_svm`, `projected_kernel_svm`), SHAP/LIME is substantially cheaper than for fidelity kernel models due to O(N) prediction cost.

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

After running an experiment, apply a saved model to new data:

```bash
claryon infer \
    --model-dir Results/tabular_binary/xgboost/seed_42/fold_0/ \
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

Available models: `xgboost`, `lightgbm`, `catboost`, `mlp`, `tabpfn`, `cnn_2d`, `cnn_3d`, `angle_pqk_svm`, `projected_kernel_svm`, `kernel_svm`, `qdc_hadamard`, `quantum_gp`, `qnn`, `qcnn_muw`, `qcnn_alt`, `tabm`, `realmlp`, `modernnca`.

---

## Notebooks

| Notebook | Content |
|---|---|
| `01_quickstart.ipynb` | Install, load iris, run XGBoost, inspect predictions |
| `02_tabular_classification.ipynb` | Full tabular workflow with multiple models |
| `03_quantum_models.ipynb` | Quantum models on iris: angle PQK SVM and amplitude-encoded models |
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
│   ├── quantum/              # Angle PQK SVM, Kernel SVM, QDC, GP, QNN, QCNN
│   ├── presets.yaml          # Complexity preset definitions
│   └── ensemble.py           # Softmax averaging
├── explainability/           # SHAP, LIME, GradCAM, plots
├── evaluation/               # Metrics, Friedman/Nemenyi, GDQ, figures
└── reporting/                # Structured LaTeX, Markdown, BibTeX

scripts/
├── run_benchmark.sh          # Primary benchmark runner for included datasets
└── run_validation.sh         # Quick validation script

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
4. Auto-discovered - no other code changes needed
5. Optionally add prose to `claryon/reporting/method_descriptions.yaml`

See notebook `08_custom_model_guide.ipynb`.

---

## Docker / Singularity

```bash
# Docker (CPU)
docker build -t claryon .
docker run -v $(pwd)/data:/app/data -v $(pwd)/Results:/app/Results \
    claryon run -c configs/example_tabular.yaml

# Docker (GPU)
docker build -f Dockerfile.gpu -t claryon-gpu .
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/Results:/app/Results \
    claryon-gpu run -c configs/example_tabular.yaml

# Singularity (HPC)
singularity build claryon.sif singularity.def
singularity run claryon.sif run -c configs/example_tabular.yaml
```

---

## Development

```bash
python -m pytest tests/ -q --timeout=300
ruff check claryon/ tests/
```

CI runs on Python 3.10-3.12 via GitHub Actions.

---

## References

- Papp L, Visvikis D, Sollini M, Shi K, Kirienko M. "The Dawn of Quantum AI in Nuclear Medicine: an EANM Perspective." *The EANM Journal*, 2026 (in revision). CLARYON is the official code repository for this manuscript (via [EANM-AI-QC](https://github.com/lpapp-muw/EANM-AI-QC)).
- Moradi S, Brandner C, Spielvogel C, Krajnc D, Hillmich S, Wille R, Drexler W, Papp L. "Clinical data classification with noisy intermediate scale quantum computers." *Scientific Reports* 12, 1851 (2022). https://doi.org/10.1038/s41598-022-05971-9
- Moradi S, Spielvogel C, Krajnc D, Brandner C, Hillmich S, Wille R, Traub-Weidinger T, Li X, Hacker M, Drexler W, Papp L. "Error mitigation enables PET radiomic cancer characterization on quantum computers." *Eur J Nucl Med Mol Imaging* 50, 3826-3837 (2023). https://doi.org/10.1007/s00259-023-06362-6
- Moradi S, et al. "Quantum Convolutional Neural Networks for Predicting ISUP Grade risk in [68Ga]Ga-PSMA Primary Prostate Cancer Patients." Under revision.
- Huang H-Y, Broughton M, Mohseni M, Babbush R, Boixo S, Neven H, McClean JR. "Power of data in quantum machine learning." *Nature Communications* 12, 2631 (2021). https://doi.org/10.1038/s41467-021-22539-9
- Thanasilp S, Wang S, Cerezo M, Holmes Z. "Exponential concentration in quantum kernel methods." *Nature Communications* 15, 5200 (2024). https://doi.org/10.1038/s41467-024-49287-w
- Shaydulin R, Wild SM. "Importance of kernel bandwidth in quantum machine learning." *Physical Review A* 106, 042407 (2022). https://doi.org/10.1103/PhysRevA.106.042407
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
