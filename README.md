# CLARYON

**CLassical-quantum AI for Reproducible Explainable OpeN-source medicine**

CLARYON is a YAML-driven machine learning framework that unifies classical, quantum, and deep learning models under a single reproducible pipeline. It supports tabular data, NIfTI medical images (PET/CT/MR), TIFF, and radiomics feature extraction — with built-in preprocessing (z-score normalization, mRMR feature selection), explainability (SHAP, LIME), statistical comparison (Friedman/Nemenyi, Geometric Difference score), and publication-ready LaTeX reporting.

**Author**: Laszlo Papp, PhD — EANM AI Committee member, Applied Quantum Computing Group, Center for Medical Physics and Biomedical Engineering, Medical University of Vienna — laszlo.papp@meduniwien.ac.at

Developed for the EANM (European Association of Nuclear Medicine) AI Committee and quantum AI researchers.

---

## Features

**Models** — 22 registered, from gradient boosting to quantum circuits:

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

**Data modalities**: Tabular CSV/Parquet, NIfTI (.nii/.nii.gz) with user-defined image and mask patterns, TIFF with metadata, legacy FDB/LDB (DEBI-NN format).

**Preprocessing**: Z-score normalization (fitted on training fold, applied to test fold), mRMR feature selection (Spearman-based redundancy clustering with configurable threshold), optional radiomics extraction (pyradiomics), image normalization (per-image or cohort-global min-max). All preprocessing state is saved per fold for reproducible inference.

**Binary grouping**: User-defined relabeling of multi-class datasets into binary classification. Specify which original labels map to positive and negative classes, enabling clinically meaningful binary endpoints (e.g., ISUP grade grouping in prostate cancer).

**Task types**: Binary classification, multi-class classification, regression. Ordinal regression and survival analysis planned.

**Explainability**: SHAP (permutation-based, works on all models including quantum), LIME, GradCAM (CNN), Integrated Gradients (planned), quantum parameter-shift attribution (planned), conformal prediction (planned).

**Evaluation**: 12 registered metrics (BACC, AUC, sensitivity, specificity, PPV, NPV, accuracy, log-loss, MAE, MSE, R², Youden's J threshold optimization), Friedman/Nemenyi statistical tests, bootstrap confidence intervals, Geometric Difference score for quantum advantage assessment.

**Reporting**: Structured methods and results LaTeX sections auto-generated from experiment config. Pre-written prose for each model, metric, and method pulled from a text registry (`method_descriptions.yaml`). Markdown reports. Ensemble performance reported alongside individual models. BibTeX references auto-collected.

**Model Presets**: Five complexity levels (quick/small/medium/large/exhaustive) plus `auto` mode that selects the highest-quality preset fitting the time budget. Per-model preset overrides, category defaults, and explicit params — composable with clear resolution priority.

**Inference**: `claryon infer` loads a saved model + preprocessing state and predicts on new data without needing the original config.

**Resource Safety**: Preflight memory/runtime checks before training. Models that would exceed 80% of available RAM are skipped gracefully with an error log. MemoryError is caught and logged, never crashes.

**Provenance**: `run_info.json` records version, git commit, timestamp, config hash, and runtime. `config_used.yaml` preserves the exact config used.

**Reproducibility**: Deterministic seeding across all stochastic operations (numpy, scikit-learn, PyTorch, PennyLane). Single YAML config defines the entire experiment. Preprocessing state (z-score coefficients, selected feature indices) saved per fold for exact inference reproduction.

---

## Installation

```bash
# Core (tabular models only)
pip install claryon

# With quantum models
pip install claryon[quantum]

# With medical imaging support
pip install claryon[imaging,radiomics]

# With boosting + explainability + reporting
pip install claryon[boosting,explain,report]

# Everything
pip install claryon[all]
```

**From source** (development):

```bash
git clone https://github.com/<org>/claryon.git
cd claryon
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

**Requirements**: Python >= 3.10. Tested on 3.10, 3.11, 3.12.

---

## Quickstart

### 1. Prepare your data

Tabular CSV with semicolon separator, a `label` column, and an optional `Key` column for sample IDs:

```
Key;f0;f1;f2;f3;label
S0000;5.1;3.5;1.4;0.2;0
S0001;7.0;3.2;4.7;1.4;1
...
```

### 2. Write a YAML config

```yaml
experiment:
  name: my_experiment
  seed: 42
  results_dir: Results/my_experiment
  complexity: medium          # quick/small/medium/large/exhaustive/auto
  max_runtime_minutes: 120    # budget for auto mode

data:
  tabular:
    path: data/my_data.csv
    label_col: label
    id_col: Key
    sep: ";"

preprocessing:
  zscore: true
  feature_selection: true
  spearman_threshold: 0.8

cv:
  strategy: kfold
  n_folds: 5
  seeds: [42, 123]

models:
  - name: xgboost
    type: tabular
    params:
      n_estimators: 200
  - name: lightgbm
    type: tabular
  - name: kernel_svm
    type: tabular_quantum
  - name: qcnn_muw
    type: tabular_quantum
    params:
      epochs: 100
      lr: 0.01

explainability:
  shap: true
  lime: true
  max_features: 32

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
├── run_info.json                    # provenance metadata
├── config_used.yaml                 # exact config used
├── xgboost/
│   ├── seed_42/fold_0/
│   │   ├── Predictions.csv
│   │   ├── preprocessing_state.json
│   │   ├── model.json               # saved model
│   │   └── model_params.json        # resolved params
│   ├── ...
│   └── explanations/
│       ├── shap_values.npy
│       ├── shap_bar.png
│       ├── shap_summary_beeswarm.png
│       ├── lime_explanations.json
│       └── lime_explanation_sample_0.png
├── kernel_svm/
│   └── ...
└── qcnn_muw/
    └── ...
```

Predictions are semicolon-separated CSVs:

```
Key;Actual;Predicted;P0;P1;Fold;Seed
S0042;0;0;0.982;0.018;0;42
S0146;1;1;0.004;0.996;0;42
```

Preprocessing state is saved per fold as JSON, storing z-score coefficients and selected feature indices for reproducible inference on new data.

---

## Preprocessing Pipeline

Preprocessing runs inside the cross-validation loop, fitting on training data only to prevent data leakage.

### Z-Score Normalization

All features are standardized to zero mean and unit variance. Parameters (mean, std) are computed on the training fold and applied to the test fold. Stored in `preprocessing_state.json`.

### mRMR Feature Selection

Minimum Redundancy Maximum Relevance: features with Spearman rank correlation above the threshold (default 0.8) are clustered as redundant. Within each cluster, the feature with the highest correlation to the target label is kept. This is critical for quantum models — reducing 306 radiomics features to ~24 drops qubit requirements from 9 to 5.

```yaml
preprocessing:
  zscore: true
  feature_selection: true
  spearman_threshold: 0.8     # features with |rho| > 0.8 are redundant
  max_features: 32            # optional hard cap
```

### Image Normalization (CNN / qCNN)

```yaml
preprocessing:
  image_normalization: per_image       # each volume scaled to [0, 1] independently
  # OR
  image_normalization: cohort_global   # global min/max from training set
```

### Binary Grouping

User-defined multi-class to binary relabeling:

```yaml
binary_grouping:
  enabled: true
  positive: [3, 4]    # ISUP grades 3+4 -> class 1 (high risk)
  negative: [1, 2]    # ISUP grades 1+2 -> class 0 (low risk)
```

---

## NIfTI / Medical Imaging

CLARYON loads NIfTI volumes with paired masks. Image and mask file patterns are user-configurable — not limited to PET.

```yaml
data:
  imaging:
    path: data/nifti_dataset
    format: nifti
    image_pattern: "*"          # match all non-mask NIfTI files (default)
    mask_pattern: "*mask*"      # glob for mask files

models:
  - name: cnn_3d
    type: imaging
    params:
      epochs: 20
      batch_size: 4
```

For radiomics extraction from NIfTI + masks (via pyradiomics):

```yaml
data:
  imaging:
    path: data/nifti_dataset
    format: nifti
    mask_pattern: "*mask*"
  radiomics:
    extract: true
    config: configs/pyradiomics_default.yaml
```

---

## Quantum Models

Quantum models use PennyLane's `default.qubit` simulator. Data is automatically amplitude-encoded (padded to the next power of 2, L2-normalized) when `type: tabular_quantum` is set. The number of qubits is derived from the encoding automatically. After mRMR reduces features, qubit requirements drop accordingly.

| Model | Circuit | Reference |
|---|---|---|
| `kernel_svm` | Amplitude embedding, Projector measurement, SVC | Havlicek et al., 2019 |
| `sq_kernel_svm` | Mottonen + adjoint Mottonen, Projector kernel, linear prediction | Moradi et al., 2022 |
| `qdc_hadamard` | Ancilla + controlled Mottonen + Hadamard test, class-max similarity | Moradi et al., 2022 |
| `qdc_swap` | Two registers + CSWAP + ancilla, class-max similarity (2n+1 qubits) | Moradi et al., 2022 |
| `quantum_gp` | Mottonen kernel, full GP posterior (mean + cov), sigmoid classification | Moradi et al., 2023 |
| `qnn` | Per-class Mottonen + Rot/CNOT layers, margin loss (PyTorch) | Moradi et al., 2023 |
| `qcnn_muw` | Amplitude embedding, conv (IsingXX/YY/ZZ + U3), pool, ArbitraryUnitary, Projector | Papp et al., under revision |
| `qcnn_alt` | Amplitude embedding, alternative conv/pool architecture, Projector | MedUni Wien design |
| `hybrid` | Quantum-classical hybrid (stub) | -- |

Practical qubit limit: <=30 (simulator). Warnings are logged above 20 qubits.

### Geometric Difference Framework (Huang et al. 2021)

Full quantum advantage assessment following Huang et al. (2021):
- **Geometric difference** g(K^C || K^Q): measures structural difference between quantum and classical kernels
- **Model complexity** s_K(N): quantifies how well the kernel fits the data
- **Effective dimension** d: rank of the quantum kernel matrix
- **Decision logic**: classical_sufficient / quantum_advantage_likely / inconclusive

Enable in config:
```yaml
evaluation:
  geometric_difference: true
```

Results saved to `Results/<experiment>/geometric_difference/` with visualization report.

---

## Configuration Reference

### Experiment

| Key | Type | Default | Description |
|---|---|---|---|
| `experiment.name` | string | `"experiment"` | Experiment identifier |
| `experiment.seed` | int | `42` | Global random seed |
| `experiment.results_dir` | string | `"Results"` | Output directory |

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
| `data.radiomics.config` | string | -- | Path to pyradiomics YAML |
| `data.fusion` | string | `"early"` | `early`, `late`, or `intermediate` |

### Preprocessing

| Key | Type | Default | Description |
|---|---|---|---|
| `preprocessing.zscore` | bool | `true` | Z-score normalize features |
| `preprocessing.feature_selection` | bool | `true` | Run mRMR feature selection |
| `preprocessing.spearman_threshold` | float | `0.8` | Redundancy threshold (0.0-1.0) |
| `preprocessing.max_features` | int | -- | Optional hard cap after mRMR |
| `preprocessing.image_normalization` | string | `"per_image"` | `per_image` or `cohort_global` |

### Binary Grouping

| Key | Type | Default | Description |
|---|---|---|---|
| `binary_grouping.enabled` | bool | `false` | Enable binary relabeling |
| `binary_grouping.positive` | list | `[]` | Original labels mapped to class 1 |
| `binary_grouping.negative` | list | `[]` | Original labels mapped to class 0 (if empty: everything not in positive) |

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
  - name: xgboost          # Registry name
    type: tabular           # tabular, tabular_quantum, or imaging
    preset: medium          # Optional: quick/small/medium/large/exhaustive
    params: {}              # Explicit params override presets
    enabled: true           # Set false to skip
```

Available model names: `xgboost`, `lightgbm`, `catboost`, `mlp`, `tabpfn`, `debinn`, `cnn_2d`, `cnn_3d`, `kernel_svm`, `sq_kernel_svm`, `qdc_hadamard`, `qdc_swap`, `quantum_gp`, `qnn`, `qcnn_muw`, `qcnn_alt`, `hybrid`, `tabm`, `realmlp`, `modernnca`.

### Explainability

| Key | Type | Default | Description |
|---|---|---|---|
| `explainability.shap` | bool | `false` | Run SHAP explanations |
| `explainability.lime` | bool | `false` | Run LIME explanations |
| `explainability.grad_cam` | bool | `false` | Run GradCAM (CNN only) |
| `explainability.max_features` | int | `32` | Feature cap for explainability |
| `explainability.max_test_samples` | int | `5` | Samples to explain |

### Evaluation

| Key | Type | Default | Description |
|---|---|---|---|
| `evaluation.metrics` | list[str] | `[bacc, auc, ...]` | Metrics to compute |
| `evaluation.statistical_tests` | list[str] | `[]` | `friedman` supported |
| `evaluation.confidence_level` | float | `0.95` | CI level |

Available metrics: `bacc`, `auc`, `sensitivity`, `specificity`, `ppv`, `npv`, `accuracy`, `logloss`, `mae`, `mse`, `r2`.

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
├── __init__.py
├── __main__.py              # python -m claryon entry point
├── cli.py                   # CLI (run, train, evaluate, explain, report, list-models)
├── config_schema.py         # Pydantic YAML config validation
├── pipeline.py              # 8-stage orchestrator
├── registry.py              # @register decorator for models, metrics, explainers
├── determinism.py           # Seed + thread control
├── io/
│   ├── base.py              # Dataset dataclass, LabelMapper, TaskType
│   ├── tabular.py           # CSV/Parquet -> Dataset
│   ├── nifti.py             # NIfTI + mask -> Dataset (user-defined patterns)
│   ├── tiff.py              # TIFF + metadata -> Dataset
│   ├── fdb_ldb.py           # Legacy DEBI-NN format
│   └── predictions.py       # Semicolon-separated Predictions.csv writer/reader
├── preprocessing/
│   ├── state.py             # PreprocessingState (save/load/apply per fold)
│   ├── tabular_prep.py      # Z-score normalization (stateful fit/apply)
│   ├── feature_selection.py # mRMR (Spearman redundancy + relevance)
│   ├── image_prep.py        # Per-image / cohort-global normalization
│   ├── radiomics.py         # PyRadiomics wrapper
│   └── splits.py            # k-fold, holdout, nested CV, GroupKFold
├── encoding/
│   ├── base.py              # QuantumEncoding ABC
│   ├── amplitude.py         # Pad to 2^n, L2-normalize
│   └── angle.py             # One qubit per feature
├── models/
│   ├── base.py              # ModelBuilder ABC, InputType, TaskType
│   ├── classical/           # XGBoost, LightGBM, CatBoost, MLP, CNN 2D/3D, ...
│   ├── quantum/             # Kernel SVM, sq-Kernel SVM, QDC, GP, QNN, QCNN, ...
│   └── ensemble.py          # Softmax averaging / mean
├── explainability/
│   ├── shap_.py             # SHAP (permutation-based)
│   ├── lime_.py             # LIME
│   ├── gradcam.py           # GradCAM for CNNs
│   └── ...                  # Integrated gradients, conformal (stubs)
├── evaluation/
│   ├── metrics.py           # 12 registered metrics + Youden's J threshold
│   ├── comparator.py        # Friedman/Nemenyi, bootstrap CI
│   ├── geometric_difference.py  # GDQ score (Huang et al. 2021)
│   ├── figures.py           # ROC, confusion matrix, CD diagram
│   └── results_store.py     # Results aggregation
└── reporting/
    ├── structured_report.py  # Prose-based methods.tex from text registry
    ├── method_descriptions.yaml  # Pre-written text blocks per model/method
    ├── references.bib        # Auto-collected BibTeX entries
    ├── latex_report.py       # Results table -> .tex
    └── markdown_report.py    # -> .md
```

---

## CLI Reference

```bash
# Run full experiment
claryon -v run -c config.yaml

# Inference on new data
claryon infer --model-dir Results/exp/xgboost/seed_42/fold_0/ \
    --input new_patients.csv --output predictions.csv

# List registered models
claryon list-models

# List registered metrics
claryon list-metrics

# Validate config without running
claryon validate-config -c config.yaml
```

Verbosity: `-v` for INFO (stage progress + summary table), `-vv` for DEBUG (per-fold logs + PennyLane traces). No flags: summary table only.

---

## Docker / Singularity

```bash
# Docker (CPU)
docker build -t claryon .
docker run -v $(pwd)/data:/app/data -v $(pwd)/Results:/app/Results claryon run -c configs/my_config.yaml

# Docker (GPU)
docker build -f Dockerfile.gpu -t claryon-gpu .
docker run --gpus all -v ... claryon-gpu run -c configs/my_config.yaml

# Singularity (HPC)
singularity build claryon.sif singularity.def
singularity run claryon.sif run -c configs/my_config.yaml
```

---

## Adding a New Model

1. Create `claryon/models/classical/mymodel_.py` (or `quantum/`)
2. Subclass `ModelBuilder` and implement `fit()`, `predict_proba()`, `predict()`
3. Decorate with `@register("model", "mymodel")`
4. Optionally add a `method_description` class attribute for auto-generated LaTeX methods sections
5. The model is auto-discovered by the pipeline — no other changes needed

```python
from __future__ import annotations
from ..base import ModelBuilder, InputType
from ...io.base import TaskType
from ...registry import register

@register("model", "mymodel")
class MyModel(ModelBuilder):
    # Optional: auto-included in methods.tex if no YAML entry exists
    method_description = "My model uses a novel approach to classification."
    method_cite_key = "Author2026"

    def __init__(self, seed: int = 42, **kwargs):
        self._seed = seed

    @property
    def name(self) -> str:
        return "mymodel"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY, TaskType.MULTICLASS)

    def fit(self, X, y, task_type, **kwargs):
        ...

    def predict_proba(self, X):
        ...  # Return (n_samples, n_classes) array
```

For built-in models, add a prose description to `claryon/reporting/method_descriptions.yaml` for high-quality structured LaTeX output.

---

## Notebooks

Tutorial notebooks are available in `examples/notebooks/`:

| Notebook | Content |
|---|---|
| `00_quickstart.ipynb` | Install, load iris, run XGBoost, inspect predictions |
| `01_tabular_demo.ipynb` | Full tabular workflow with multiple models |
| `02_quantum_models.ipynb` | All quantum models on iris binary |
| `03_nifti_imaging.ipynb` | NIfTI + masks -> 3D CNN |
| `04_explainability.ipynb` | SHAP + LIME on classical and quantum models |
| `05_results_dashboard.ipynb` | Metrics visualization and statistical comparison |
| `06_radiomics.ipynb` | Radiomics extraction -> merge -> train |
| `07_custom_model.ipynb` | How to add a new model using @register |

---

## Development

```bash
# Run tests
python -m pytest tests/ -q --timeout=300

# Run with coverage
python -m pytest tests/ --cov=claryon --cov-report=html

# Lint
ruff check claryon/ tests/
```

298+ tests across unit, integration, and model smoke tests. CI runs on Python 3.10-3.12 via GitHub Actions.

---

## Origin

CLARYON absorbs two existing codebases:

1. **EANM-AI-QC** (v0.8.0) — Quantum ML framework for nuclear medicine developed at MedUni Wien. PennyLane QCNN, quantum kernel SVM, SHAP/LIME explainability, NIfTI + tabular I/O.

2. **DEBI-NN Benchmark Harness** — Tabular classification benchmark across 28 datasets with 8 classical competitor methods, statistical analysis, and LaTeX reporting.

**Two-repo strategy**: This repository (CLARYON) contains all code, tests, and CI. The [EANM-AI-QC repository](https://github.com/<org>/eanm-ai-qc) is retained as an educational hub for the EANM AI Committee (teaching materials, tutorials, committee documentation).

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
  title        = {{CLARYON}: Classical-quantum {AI} for Reproducible Explainable Open-source Medicine},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/<org>/claryon}
}
```

---

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.

Compatible with academic research use, community contributions, and integration with GPLv2/v3-licensed dependencies (e.g., pyradiomics).

---

## Acknowledgments

Medical University of Vienna (MedUni Wien), EANM AI Committee, MORPHEDRON.
