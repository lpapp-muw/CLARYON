# CLARYON

**CLassical-quantum AI for Reproducible Explainable OpeN-source medicine**

CLARYON is a YAML-driven machine learning framework that unifies classical, quantum, and deep learning models under a single reproducible pipeline. It supports tabular data, NIfTI medical images (PET/CT/MR), TIFF (OCT/biophotonics), and radiomics feature extraction — with built-in explainability (SHAP, LIME), statistical comparison (Friedman/Nemenyi), and publication-ready LaTeX reporting.

Developed for the EANM (European Association of Nuclear Medicine) AI Committee, OCT/biophotonics research groups, and quantum AI researchers.

---

## Features

**Models** — 17 registered, from gradient boosting to quantum circuits:

| Category | Models | Backend |
|---|---|---|
| Gradient boosting | XGBoost, LightGBM, CatBoost | scikit-learn API |
| Neural networks | MLP, 2D CNN, 3D CNN | scikit-learn / PyTorch |
| Quantum ML | Quantum kernel SVM, QCNN-MUW, QCNN-ALT, VQC | PennyLane |
| Quantum-classical | Hybrid model | PennyLane + PyTorch |
| Ensemble | Softmax averaging (classification), mean (regression) | numpy |

**Data modalities**: Tabular CSV/Parquet, NIfTI (.nii/.nii.gz) with masks, TIFF with metadata, DICOM (planned), legacy FDB/LDB (DEBI-NN format).

**Task types**: Binary classification, multi-class classification, regression. Ordinal regression and survival analysis planned.

**Explainability**: SHAP (permutation-based, works on all models including quantum), LIME, GradCAM (CNN), Integrated Gradients (planned), quantum parameter-shift attribution (planned), conformal prediction (planned).

**Evaluation**: 12 registered metrics (BACC, AUC, sensitivity, specificity, PPV, NPV, accuracy, log-loss, MAE, MSE, R², Youden's J threshold optimization), Friedman/Nemenyi statistical tests, bootstrap confidence intervals.

**Reporting**: Markdown and LaTeX report generation from experiment results. Methods section + results table with metrics, auto-generated from the YAML config.

**Reproducibility**: Deterministic seeding across all stochastic operations (numpy, scikit-learn, PyTorch, PennyLane). Single YAML config defines the entire experiment.

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

**Requirements**: Python ≥ 3.10. Tested on 3.10, 3.11, 3.12.

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

data:
  tabular:
    path: data/my_data.csv
    label_col: label
    id_col: Key
    sep: ";"

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
      epochs: 15
      lr: 0.02

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
├── xgboost/
│   ├── seed_42/fold_0/Predictions.csv
│   ├── ...
│   └── explanations/
│       ├── shap_values.npy
│       └── lime_explanations.json
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

---

## NIfTI / Medical Imaging

CLARYON loads NIfTI volumes with paired masks for nuclear medicine (PET/CT/MR), neuroimaging, and general medical imaging workflows.

```yaml
data:
  imaging:
    path: data/nifti_dataset    # must have Train/ and Test/ subdirs
    format: nifti
    mask_pattern: "*mask*"

models:
  - name: cnn_3d
    type: imaging
    params:
      epochs: 20
      batch_size: 4
```

The pipeline automatically pairs `*_PET.nii.gz` with `*_mask.nii.gz`, applies masks (zeroing non-ROI voxels), and either flattens for tabular models or passes 5D tensors to CNNs.

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

Quantum models use PennyLane's `default.qubit` simulator. Data is automatically amplitude-encoded (padded to the next power of 2, L2-normalized) when `type: tabular_quantum` is set in the config. The number of qubits is derived from the encoding automatically.

| Model | Circuit | Reference |
|---|---|---|
| `kernel_svm` | Amplitude embedding → Projector measurement → `\|⟨x\|y⟩\|²` kernel → SVC | Havlíček et al., 2019 |
| `qcnn_muw` | Amplitude embedding → conv (IsingXX/YY/ZZ + U3) → pool (controlled Rot) → ArbitraryUnitary → Projector | MedUni Wien design |
| `qcnn_alt` | Amplitude embedding → alternative conv/pool architecture → Projector | MedUni Wien design |
| `vqc` | Variational quantum classifier (stub) | — |
| `hybrid` | Quantum-classical hybrid (stub) | — |

Practical qubit limit: ≤30 (simulator). Warnings are logged above 20 qubits.

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
| `data.imaging.path` | string | — | Path to imaging directory |
| `data.imaging.format` | string | `"nifti"` | `nifti`, `tiff`, or `dicom` |
| `data.imaging.mask_pattern` | string | `"*mask*"` | Glob for mask files |
| `data.radiomics.extract` | bool | `false` | Run pyradiomics extraction |
| `data.radiomics.config` | string | — | Path to pyradiomics YAML |
| `data.fusion` | string | `"early"` | `early`, `late`, or `intermediate` |

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
    params: {}              # Passed to model constructor
    enabled: true           # Set false to skip
```

Available model names: `xgboost`, `lightgbm`, `catboost`, `mlp`, `tabpfn`, `debinn`, `cnn_2d`, `cnn_3d`, `kernel_svm`, `qcnn_muw`, `qcnn_alt`, `vqc`, `hybrid`, `tabm`, `realmlp`, `modernnca`.

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
├── pipeline.py              # 7-stage orchestrator
├── registry.py              # @register decorator for models, metrics, explainers
├── determinism.py           # Seed + thread control
├── io/
│   ├── base.py              # Dataset dataclass, LabelMapper, TaskType
│   ├── tabular.py           # CSV/Parquet → Dataset
│   ├── nifti.py             # NIfTI + mask → Dataset
│   ├── tiff.py              # TIFF + metadata → Dataset
│   ├── fdb_ldb.py           # Legacy DEBI-NN format
│   └── predictions.py       # Semicolon-separated Predictions.csv writer/reader
├── preprocessing/
│   ├── tabular_prep.py      # Imputation, scaling, encoding
│   ├── image_prep.py        # Resampling, normalization
│   ├── radiomics.py         # PyRadiomics wrapper
│   └── splits.py            # k-fold, holdout, nested CV, GroupKFold
├── encoding/
│   ├── base.py              # QuantumEncoding ABC
│   ├── amplitude.py         # Pad to 2^n, L2-normalize
│   └── angle.py             # One qubit per feature
├── models/
│   ├── base.py              # ModelBuilder ABC, InputType, TaskType
│   ├── classical/           # XGBoost, LightGBM, CatBoost, MLP, CNN 2D/3D, ...
│   ├── quantum/             # Kernel SVM, QCNN MUW/ALT, VQC, Hybrid
│   └── ensemble.py          # Softmax averaging / mean
├── explainability/
│   ├── shap_.py             # SHAP (permutation-based)
│   ├── lime_.py             # LIME
│   ├── gradcam.py           # GradCAM for CNNs
│   └── ...                  # Integrated gradients, conformal (stubs)
├── evaluation/
│   ├── metrics.py           # 12 registered metrics + Youden's J threshold
│   ├── comparator.py        # Friedman/Nemenyi, bootstrap CI
│   ├── figures.py           # ROC, confusion matrix, CD diagram
│   └── results_store.py     # Results aggregation
└── reporting/
    ├── latex_report.py       # Jinja2 → .tex
    └── markdown_report.py    # Jinja2 → .md
```

---

## CLI Reference

```bash
# Run full experiment
claryon -v run -c config.yaml

# List registered models
claryon list-models

# List registered metrics
claryon list-metrics

# Validate config without running
claryon validate-config -c config.yaml
```

Verbosity: `-v` for INFO, `-vv` for DEBUG (includes PennyLane circuit traces).

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
4. The model is auto-discovered by the pipeline — no other changes needed

```python
from __future__ import annotations
from ..base import ModelBuilder, InputType
from ...io.base import TaskType
from ...registry import register

@register("model", "mymodel")
class MyModel(ModelBuilder):
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

165 tests across unit, integration, and model smoke tests. CI runs on Python 3.10–3.12 via GitHub Actions.

---

## Origin

CLARYON absorbs two existing codebases:

1. **EANM-AI-QC** (v0.8.0) — Quantum ML framework for nuclear medicine developed at MedUni Wien. PennyLane QCNN, quantum kernel SVM, SHAP/LIME explainability, NIfTI + tabular I/O.

2. **DEBI-NN Benchmark Harness** — Tabular classification benchmark across 28 datasets with 8 classical competitor methods, statistical analysis, and LaTeX reporting.

The combined framework supports non-nuclear-medicine domains including optical coherence tomography (OCT), photoacoustics, biophotonics, and general medical imaging research.

**Two-repo strategy**: This repository (CLARYON) contains all code, tests, and CI. The [EANM-AI-QC repository](https://github.com/<org>/eanm-ai-qc) is retained as an educational hub for the EANM AI Committee (teaching materials, tutorials, committee documentation).

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

MedUni Wien, EANM AI Committee, MORPHEDRON.
