# CLARYON

**The EANM AI Committee's git repository to support Nuclear Medicine research**

**CLassical-quantum AI for Reproducible Explainable OpeN-source medicine**

CLARYON is a YAML-driven machine learning framework that unifies classical, quantum, and deep learning models under a single reproducible pipeline. It supports tabular data, NIfTI medical images (PET/CT/MR), TIFF, and radiomics feature extraction — with built-in preprocessing (z-score normalization, mRMR feature selection), explainability (SHAP, LIME), statistical comparison (Friedman/Nemenyi, Geometric Difference score), and publication-ready LaTeX reporting.

**Author**: Laszlo Papp, PhD — EANM AI Committee member, Applied Quantum Computing Group, Center for Medical Physics and Biomedical Engineering, Medical University of Vienna — laszlo.papp@meduniwien.ac.at

---

## Features

**Models** — 21 registered, from gradient boosting to quantum circuits:

| Category | Models | Backend |
|---|---|---|
| Gradient boosting | XGBoost, LightGBM, CatBoost | scikit-learn API |
| Neural networks | MLP, 2D CNN, 3D CNN | scikit-learn / PyTorch |
| Quantum ML | Quantum kernel SVM, Simplified quantum kernel SVM, QCNN-MUW, QCNN-ALT, QNN | PennyLane |
| Quantum distance | Hadamard distance classifier, SWAP distance classifier | PennyLane |
| Quantum GP | Quantum Gaussian Process | PennyLane |
| Quantum-classical | Hybrid model | PennyLane + PyTorch |
| Ensemble | Softmax averaging (classification), mean (regression) | numpy |
| Evaluation | Geometric Difference score (GDQ) | numpy / scipy |

**Data modalities**: Tabular CSV/Parquet, NIfTI (.nii/.nii.gz) with user-defined image and mask patterns, TIFF with metadata.

**Preprocessing**: Z-score normalization (fitted on training fold, applied to test fold — automatically skipped for quantum models), mRMR feature selection (Spearman-based redundancy clustering), optional radiomics extraction (pyradiomics), image normalization (per-image or cohort-global min-max). All preprocessing state is saved per fold for reproducible inference.

**Binary grouping**: User-defined relabeling of multi-class datasets into binary classification for clinically meaningful endpoints (e.g., ISUP grade grouping in prostate cancer).

**Explainability**: SHAP (permutation-based, works on all models including quantum), LIME. Generates PNG plots (beeswarm, bar, per-sample waterfall/explanation).

**Evaluation**: 12 registered metrics (BACC, AUC, sensitivity, specificity, PPV, NPV, accuracy, log-loss, MAE, MSE, R-squared, Youden's J threshold optimization), Friedman/Nemenyi statistical tests, bootstrap confidence intervals, Geometric Difference score for quantum advantage assessment.

**Reporting**: Structured methods and results LaTeX sections auto-generated from experiment config with pre-written prose from a text registry. Markdown reports. Mean +/- std for all metrics. BibTeX references auto-collected.

**Reproducibility**: Deterministic seeding across all stochastic operations. Single YAML config defines the entire experiment. Preprocessing state saved per fold. Model complexity presets (quick/small/medium/large/exhaustive/auto) for consistent, publishable results.

---

## Installation

### Core install

```bash
pip install claryon
```

### With optional dependencies

```bash
# Quantum models (PennyLane)
pip install claryon[quantum]

# Medical imaging (NIfTI, TIFF)
pip install claryon[imaging]

# Radiomics extraction
pip install numpy versioneer
pip install pyradiomics==3.0.1 --no-build-isolation
pip install claryon[radiomics]

# Gradient boosting (XGBoost, LightGBM, CatBoost)
pip install claryon[boosting]

# Explainability (SHAP, LIME)
pip install claryon[explain]

# Reporting (LaTeX, Markdown, figures)
pip install claryon[report]

# Deep learning (PyTorch CNNs)
pip install claryon[torch]

# TabPFN (downloads pretrained weights on first use, ~500 MB)
pip install claryon[tabular-dl]

# Everything
pip install claryon[all]
```

### From source (development)

```bash
git clone https://github.com/lpapp-muw/CLARYON.git
cd CLARYON
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

### Notes on specific dependencies

**pyradiomics**: Standard `pip install pyradiomics` fails due to broken build metadata. Use: `pip install numpy versioneer && pip install pyradiomics==3.0.1 --no-build-isolation`

**TabPFN**: Downloads pretrained model weights (~500 MB) on first use. Requires Python <= 3.11.

**PyTorch**: CPU-only is sufficient for quantum models and small CNNs. For large imaging datasets, install with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Requirements

Python >= 3.10. Tested on 3.10, 3.11, 3.12.

---

## Quickstart

### 1. Prepare your data

Tabular CSV with semicolon separator, a `label` column, and an optional `Key` column:

```
Key;f0;f1;f2;f3;label
S0000;5.1;3.5;1.4;0.2;0
S0001;7.0;3.2;4.7;1.4;1
```

### 2. Write a YAML config

```yaml
experiment:
  name: my_experiment
  seed: 42
  results_dir: Results/my_experiment
  complexity: medium          # or: quick, small, large, exhaustive, auto

data:
  tabular:
    path: data/my_data.csv
    label_col: label
    id_col: Key
    sep: ";"

preprocessing:
  zscore: true                # auto-skipped for quantum models
  feature_selection: true
  spearman_threshold: 0.8

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

### 3. Run

```bash
claryon -v run -c config.yaml
```

### 4. Results

```
Results/my_experiment/
├── metrics_summary.csv
├── report.md
├── methods.tex
├── results.tex
├── references_needed.txt
├── run_info.json
├── config_used.yaml
├── xgboost/
│   ├── seed_42/fold_0/
│   │   ├── Predictions.csv
│   │   └── preprocessing_state.json
│   └── explanations/
│       ├── shap_values.npy
│       ├── shap_bar.png
│       ├── shap_waterfall_sample_0.png
│       └── lime_explanation_sample_0.png
├── kernel_svm/
│   └── ...
└── qcnn_muw/
    └── ...
```

---

## Model Complexity Presets

Nuclear medicine researchers unfamiliar with quantum model tuning should use presets instead of raw parameters. Quantum models are sensitive to training configuration — using `quick` defaults for publication will produce misleadingly poor quantum performance.

```yaml
experiment:
  complexity: medium          # recommended for most studies
```

| Preset | Quantum epochs | Quantum LR | Classical estimators | Use case |
|---|---|---|---|---|
| `quick` | 5 | 0.05 | 50 | CI/testing only. NOT for publication. |
| `small` | 30 | 0.02 | 200 | Fast exploratory analysis |
| `medium` | 100 | 0.01 | 500 | **Recommended for most studies** |
| `large` | 300 | 0.005 | 1000 | Final results for publications |
| `exhaustive` | 500 | 0.002 | 2000 | Maximum effort |
| `auto` | dataset-dependent | dataset-dependent | dataset-dependent | CLARYON analyzes your data and time budget, picks the best preset per model |

Auto mode uses `max_runtime_minutes` (default: 120) to select the highest quality preset that fits your time budget:

```yaml
experiment:
  complexity: auto
  max_runtime_minutes: 60
```

Per-model override (expert users):

```yaml
models:
  - name: qcnn_muw
    type: tabular_quantum
    preset: large             # override global complexity for this model
  - name: xgboost
    type: tabular
    params:                   # or use raw params (highest priority)
      n_estimators: 1000
```

---

## Preprocessing Pipeline

Preprocessing runs inside the cross-validation loop, fitting on training data only to prevent data leakage.

### Z-Score Normalization

Features are standardized to zero mean and unit variance. Parameters are computed on the training fold and applied to the test fold.

**Important**: Z-score is automatically skipped for quantum models. Amplitude encoding already L2-normalizes the feature vector, and prior z-score normalization distorts the geometric structure in Hilbert space, degrading quantum kernel performance by 30-40%.

### mRMR Feature Selection

Features with Spearman rank correlation above the threshold (default 0.8) are clustered as redundant. Within each cluster, the feature with the highest correlation to the target label is kept. Applied to both classical and quantum models.

```yaml
preprocessing:
  zscore: true                # auto-skipped for quantum models
  feature_selection: true
  spearman_threshold: 0.8
  max_features: 32            # optional hard cap
```

### Binary Grouping

User-defined multi-class to binary relabeling:

```yaml
binary_grouping:
  enabled: true
  positive: [3, 4]            # e.g., ISUP grades 3+4 -> high risk
  negative: [1, 2]            # e.g., ISUP grades 1+2 -> low risk
```

### Image Normalization

```yaml
preprocessing:
  image_normalization: per_image       # each volume scaled to [0, 1]
  # OR
  image_normalization: cohort_global   # global min/max from training set
```

---

## Benchmark Datasets

CLARYON includes a downloader for 12 curated medical datasets:

```bash
python -m claryon.benchmark.download_benchmark_datasets --output-dir benchmarking/datasets
```

| Dataset | Source | Samples | Features | Domain |
|---|---|---|---|---|
| wisconsin-breast-cancer | UCI | 699 | 9 | Oncology |
| heart-failure | UCI | 299 | 12 | Cardiology |
| cervical-cancer | UCI | 858 | 36 | Oncology |
| chronic-kidney-disease | UCI | 400 | 24 | Nephrology |
| spect-heart | UCI | 267 | 22 | Nuclear medicine (SPECT) |
| mammographic-mass | UCI | 961 | 5 | Radiology |
| blood-transfusion | OpenML | 748 | 4 | Hematology |
| diabetes | OpenML | 768 | 8 | Endocrinology |
| iris | OpenML | 150 | 4 | Demo/smoke test |
| hcc-survival | Kaggle | 165 | 49 | Oncology (HCC) |
| stroke-prediction | Kaggle | 5110 | 11 | Neurology |
| fetal-health | Kaggle | 2126 | 21 | Obstetrics |

Kaggle datasets require a Kaggle API token (`~/.kaggle/kaggle.json`). Use `--skip-kaggle` to skip them.

---

## NIfTI / Medical Imaging

Image and mask file patterns are user-configurable:

```yaml
data:
  imaging:
    path: data/nifti_dataset
    format: nifti
    image_pattern: "*"              # match all non-mask NIfTI files
    mask_pattern: "*mask*"

models:
  - name: cnn_3d
    type: imaging
    params:
      epochs: 20
      batch_size: 4
```

Note: CNN models (`cnn_2d`, `cnn_3d`) require imaging data. They will be skipped if only tabular data is provided. Similarly, tabular and quantum models require tabular data and will be skipped on imaging-only configs.

For radiomics extraction:

```yaml
data:
  radiomics:
    extract: true
    config: configs/pyradiomics_default.yaml
```

---

## Quantum Models

Quantum models use PennyLane's `default.qubit` simulator. Data is automatically amplitude-encoded when `type: tabular_quantum` is set. Qubit count is derived from feature count after mRMR selection.

| Model | Circuit | Reference |
|---|---|---|
| `kernel_svm` | Amplitude embedding, Projector kernel, SVC | Havlicek et al., 2019 |
| `sq_kernel_svm` | Mottonen + adjoint Mottonen, Projector kernel, linear prediction | Moradi et al., 2022 |
| `qdc_hadamard` | Ancilla + controlled Mottonen + Hadamard test, class-max similarity | Moradi et al., 2022 |
| `qdc_swap` | Two registers + CSWAP + ancilla, class-max similarity (2n+1 qubits) | Moradi et al., 2022 |
| `quantum_gp` | Mottonen kernel, full GP posterior, sigmoid classification | Moradi et al., 2023 |
| `qnn` | Per-class Mottonen + Rot/CNOT layers, margin loss (PyTorch) | Moradi et al., 2023 |
| `qcnn_muw` | Amplitude embedding, conv/pool layers, ArbitraryUnitary, Projector | Papp et al., under revision |
| `qcnn_alt` | Alternative conv/pool architecture, Projector | MedUni Wien design |
| `hybrid` | Quantum-classical hybrid (stub) | -- |

Practical qubit limit: <=20 recommended, <=30 possible. Resource warnings are logged automatically. CLARYON estimates memory and runtime before training and skips models that would exceed available resources.

### Geometric Difference Score

The GDQ score quantifies whether a quantum kernel provides a structurally different similarity measure from classical kernels (Huang et al., 2021). GDQ > 1.0 suggests potential quantum advantage; GDQ <= 1.0 means classical models are likely sufficient.

---

## Inference on New Data

After training, use saved models and preprocessing state to predict on new patients:

```bash
claryon infer \
    --model-dir Results/my_experiment/xgboost/seed_42/fold_0/ \
    --input data/new_patients.csv \
    --output predictions_new.csv
```

The inference command loads the saved model and preprocessing state (z-score coefficients, feature selection mask) and applies them to new data identically to training.

---

## Runtime Expectations

Approximate runtimes for `complexity: medium` on a single CPU core:

| Model | 150 samples, 4 features | 500 samples, 30 features | 1000 samples, 100 features |
|---|---|---|---|
| XGBoost | 1 second | 5 seconds | 15 seconds |
| LightGBM | 1 second | 3 seconds | 10 seconds |
| CatBoost | 2 seconds | 10 seconds | 30 seconds |
| MLP | 1 second | 5 seconds | 15 seconds |
| kernel_svm (quantum) | 1 minute | 15 minutes | 2+ hours |
| qcnn_muw (quantum) | 8 minutes | 1 hour | 5+ hours |
| qnn (quantum) | 5 minutes | 45 minutes | 3+ hours |
| cnn_3d | 2 minutes | 10 minutes | 30 minutes (GPU recommended) |

Times are per fold. Multiply by n_folds x n_seeds for total. GPU accelerates CNNs only; quantum models run on CPU (PennyLane simulator).

---

## Configuration Reference

### Experiment

| Key | Type | Default | Description |
|---|---|---|---|
| `experiment.name` | string | `"experiment"` | Experiment identifier |
| `experiment.seed` | int | `42` | Global random seed |
| `experiment.results_dir` | string | `"Results"` | Output directory |
| `experiment.complexity` | string | `"medium"` | `quick`, `small`, `medium`, `large`, `exhaustive`, `auto` |
| `experiment.max_runtime_minutes` | int | `120` | Time budget for auto mode |

### Data

| Key | Type | Default | Description |
|---|---|---|---|
| `data.tabular.path` | string | required | Path to CSV/Parquet |
| `data.tabular.label_col` | string | `"label"` | Label column name |
| `data.tabular.id_col` | string | `"Key"` | Sample ID column |
| `data.tabular.sep` | string | `";"` | CSV separator |
| `data.imaging.path` | string | -- | Path to imaging directory |
| `data.imaging.format` | string | `"nifti"` | `nifti` or `tiff` |
| `data.imaging.image_pattern` | string | `"*"` | Glob for image volumes |
| `data.imaging.mask_pattern` | string | `"*mask*"` | Glob for mask files |
| `data.radiomics.extract` | bool | `false` | Run pyradiomics extraction |
| `data.fusion` | string | `"early"` | `early`, `late`, or `intermediate` |

### Preprocessing

| Key | Type | Default | Description |
|---|---|---|---|
| `preprocessing.zscore` | bool | `true` | Z-score normalize (auto-skipped for quantum) |
| `preprocessing.feature_selection` | bool | `true` | Run mRMR feature selection |
| `preprocessing.spearman_threshold` | float | `0.8` | Redundancy threshold |
| `preprocessing.max_features` | int | -- | Optional hard cap after mRMR |
| `preprocessing.image_normalization` | string | `"per_image"` | `per_image` or `cohort_global` |

### Cross-Validation

| Key | Type | Default | Description |
|---|---|---|---|
| `cv.strategy` | string | `"kfold"` | `kfold`, `holdout`, `nested`, `external`, `group_kfold` |
| `cv.n_folds` | int | `5` | Number of folds |
| `cv.seeds` | list[int] | `[42]` | Seeds for repeated CV |
| `cv.test_size` | float | `0.2` | Holdout test fraction |

### Models

```yaml
models:
  - name: xgboost             # registry name
    type: tabular              # tabular, tabular_quantum, or imaging
    preset: medium             # optional: override global complexity
    params: {}                 # optional: override preset (expert)
    enabled: true              # set false to skip
```

Available: `xgboost`, `lightgbm`, `catboost`, `mlp`, `tabpfn`, `cnn_2d`, `cnn_3d`, `kernel_svm`, `sq_kernel_svm`, `qdc_hadamard`, `qdc_swap`, `quantum_gp`, `qnn`, `qcnn_muw`, `qcnn_alt`, `hybrid`, `tabm`, `realmlp`, `modernnca`.

### Explainability

| Key | Type | Default | Description |
|---|---|---|---|
| `explainability.shap` | bool | `false` | Run SHAP explanations |
| `explainability.lime` | bool | `false` | Run LIME explanations |
| `explainability.grad_cam` | bool | `false` | Run GradCAM (CNN only) |
| `explainability.max_features` | int | `32` | Feature cap |
| `explainability.max_test_samples` | int | `5` | Samples to explain |

### Evaluation

| Key | Type | Default | Description |
|---|---|---|---|
| `evaluation.metrics` | list | `[bacc, auc, ...]` | Metrics to compute |
| `evaluation.statistical_tests` | list | `[]` | `friedman` supported |
| `evaluation.confidence_level` | float | `0.95` | CI level |

### Reporting

| Key | Type | Default | Description |
|---|---|---|---|
| `reporting.markdown` | bool | `true` | Generate Markdown report |
| `reporting.latex` | bool | `false` | Generate LaTeX report |
| `reporting.figures` | bool | `true` | Generate figures |
| `reporting.figure_dpi` | int | `300` | Figure resolution |

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
│   ├── preset_resolver.py    # Preset resolution logic
│   ├── auto_complexity.py    # Auto mode (dataset + budget analysis)
│   └── ensemble.py           # Softmax averaging
├── explainability/           # SHAP, LIME, GradCAM, plots
├── evaluation/               # Metrics, Friedman/Nemenyi, GDQ, figures
├── reporting/                # Structured LaTeX, Markdown, method descriptions, BibTeX
└── benchmark/                # Dataset downloader (12 medical datasets)
```

---

## Notebooks

Tutorial notebooks in `examples/notebooks/`:

| Notebook | Content |
|---|---|
| `01_quickstart.ipynb` | Install, load iris, run XGBoost, inspect predictions |
| `02_tabular_classification.ipynb` | Full tabular workflow with multiple models |
| `03_quantum_models.ipynb` | All quantum models on iris binary |
| `04_nifti_imaging.ipynb` | NIfTI + masks, 3D CNN |
| `05_explainability.ipynb` | SHAP + LIME on classical and quantum models |
| `06_results_dashboard.ipynb` | Metrics visualization and statistical comparison |
| `07_radiomics.ipynb` | Radiomics extraction, merge, train |
| `08_custom_model_guide.ipynb` | How to add a new model using @register |

---

## Adding a New Model

1. Create `claryon/models/classical/mymodel_.py` (or `quantum/`)
2. Subclass `ModelBuilder`, implement `fit()`, `predict_proba()`
3. Decorate with `@register("model", "mymodel")`
4. The model is auto-discovered — no other code changes needed
5. Optionally add a prose description to `claryon/reporting/method_descriptions.yaml`

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
python -m pytest tests/ --cov=claryon --cov-report=html
ruff check claryon/ tests/
```

CI runs on Python 3.10-3.12 via GitHub Actions.

---

## References

- Moradi S, Brandner C, Spielvogel C, Krajnc D, Hillmich S, Wille R, Drexler W, Papp L. "Clinical data classification with noisy intermediate scale quantum computers." *Scientific Reports* 12, 1851 (2022). https://doi.org/10.1038/s41598-022-05971-9
- Moradi S, Spielvogel C, Krajnc D, Brandner C, Hillmich S, Wille R, Traub-Weidinger T, Li X, Hacker M, Drexler W, Papp L. "Error mitigation enables PET radiomic cancer characterization on quantum computers." *Eur J Nucl Med Mol Imaging* 50, 3826-3837 (2023). https://doi.org/10.1007/s00259-023-06362-6
- Papp L, et al. "Quantum Convolutional Neural Networks for Predicting ISUP Grade risk in [68Ga]Ga-PSMA Primary Prostate Cancer Patients." Under revision.
- Huang H-Y, Broughton M, Mohseni M, Babbush R, Boixo S, Neven H, McClean JR. "Power of data in quantum machine learning." *Nature Communications* 12, 2631 (2021). https://doi.org/10.1038/s41467-021-22539-9

---

## Citation

```bibtex
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

Medical University of Vienna (MedUni Wien), EANM AI Committee, MORPHEDRON.
