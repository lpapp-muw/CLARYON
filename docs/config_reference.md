# CLARYON Configuration Reference

Complete reference for all YAML configuration keys. Every CLARYON experiment is defined by a single YAML file passed to `claryon run -c <config.yaml>`.

---

## experiment

Top-level experiment settings.

| Key                  | Type     | Default        | Description                                                              |
|----------------------|----------|----------------|--------------------------------------------------------------------------|
| `name`               | `str`    | `"experiment"` | Experiment name. Used as subdirectory in results_dir.                    |
| `seed`               | `int`    | `42`           | Global random seed for reproducibility.                                  |
| `results_dir`        | `str`    | `"Results"`    | Root directory for all output files.                                     |
| `complexity`         | `str`    | `"medium"`     | Global preset level. One of: `quick`, `small`, `medium`, `large`, `exhaustive`, `auto`. |
| `max_runtime_minutes`| `int`    | `120`          | Time budget in minutes. Used by `auto` complexity mode. Minimum: 1.      |

**Example:**
```yaml
experiment:
  name: prostate_study
  seed: 42
  results_dir: Results
  complexity: medium
  max_runtime_minutes: 120
```

---

## data

Data source configuration. Supports tabular CSV and NIfTI imaging data.

| Key             | Type     | Default  | Description                                                                |
|-----------------|----------|----------|----------------------------------------------------------------------------|
| `type`          | `str`    | required | Data type: `tabular` or `imaging`.                                         |
| `path`          | `str`    | required | Path to data file (CSV) or directory (NIfTI).                              |
| `target`        | `str`    | required | Name of the target/label column in the CSV.                                |
| `separator`     | `str`    | `";"`    | CSV column separator.                                                      |
| `id_column`     | `str`    | `null`   | Column name for patient/sample IDs. Excluded from features.                |
| `feature_columns` | `list[str]` | `null` | Explicit list of feature columns. If null, all non-target/non-ID columns are used. |

### data (imaging-specific keys)

| Key              | Type     | Default  | Description                                                               |
|------------------|----------|----------|---------------------------------------------------------------------------|
| `image_dir`      | `str`    | required | Directory containing NIfTI files.                                         |
| `label_file`     | `str`    | required | CSV mapping filenames to labels.                                          |
| `image_column`   | `str`    | `"filename"` | Column in label_file with image filenames.                           |
| `target`         | `str`    | required | Column in label_file with class labels.                                   |
| `modality`       | `str`    | `"2d"`   | Image modality: `2d` or `3d`.                                             |
| `target_shape`   | `list[int]` | `null` | Resize images to this shape. E.g., `[64, 64]` for 2D, `[64, 64, 64]` for 3D. |

**Example (tabular):**
```yaml
data:
  type: tabular
  path: data/patients.csv
  target: diagnosis
  separator: ";"
  id_column: patient_id
```

**Example (imaging):**
```yaml
data:
  type: imaging
  image_dir: data/nifti/
  label_file: data/labels.csv
  image_column: filename
  target: grade
  modality: 3d
  target_shape: [64, 64, 64]
```

---

## preprocessing

Data preprocessing steps applied before model training.

| Key                    | Type     | Default  | Description                                                         |
|------------------------|----------|----------|---------------------------------------------------------------------|
| `z_score`              | `bool`   | `true`   | Standardize features to zero mean and unit variance.                |
| `feature_selection`    | `object` | `null`   | Feature selection configuration (see sub-keys below).               |
| `binary_grouping`      | `object` | `null`   | Multi-class to binary grouping (see sub-keys below).                |
| `image_normalization`  | `object` | `null`   | Image preprocessing settings (see sub-keys below).                  |

### preprocessing.feature_selection

| Key              | Type     | Default  | Description                                                            |
|------------------|----------|----------|------------------------------------------------------------------------|
| `method`         | `str`    | `"mrmr"` | Feature selection method. Currently supported: `mrmr`.                 |
| `max_features`   | `int`    | `null`   | Maximum number of features to select. If null, uses all features.      |
| `n_features`     | `int`    | `null`   | Exact number of features to select. Overrides max_features if both set.|

### preprocessing.binary_grouping

| Key              | Type        | Default  | Description                                                         |
|------------------|-------------|----------|---------------------------------------------------------------------|
| `enabled`        | `bool`      | `false`  | Enable binary grouping of multi-class targets.                      |
| `positive_classes` | `list[Any]` | required | List of class labels to treat as the positive class.              |

### preprocessing.image_normalization

| Key              | Type     | Default  | Description                                                            |
|------------------|----------|----------|------------------------------------------------------------------------|
| `method`         | `str`    | `"minmax"` | Normalization method: `minmax`, `zscore`, or `percentile`.          |
| `clip_percentile`| `float`  | `99.5`   | Upper percentile for clipping (used with `percentile` method).       |

**Example:**
```yaml
preprocessing:
  z_score: true
  feature_selection:
    method: mrmr
    max_features: 16
  binary_grouping:
    enabled: true
    positive_classes: [1, 2]
```

---

## cv

Cross-validation configuration.

| Key         | Type        | Default         | Description                                                         |
|-------------|-------------|-----------------|---------------------------------------------------------------------|
| `strategy`  | `str`       | `"stratified_kfold"` | CV strategy: `stratified_kfold`, `kfold`, `repeated_stratified_kfold`, `holdout`. |
| `n_folds`   | `int`       | `5`             | Number of folds (ignored for `holdout`).                            |
| `seeds`     | `list[int]` | `[42]`          | List of random seeds for repeated experiments. Each seed produces a full CV run. |
| `test_size` | `float`     | `0.2`           | Test set fraction (used only with `holdout` strategy).              |

**Example:**
```yaml
cv:
  strategy: stratified_kfold
  n_folds: 5
  seeds: [42, 123]
```

---

## models

List of models to train. Each entry is a model configuration object.

| Key       | Type     | Default    | Description                                                              |
|-----------|----------|------------|--------------------------------------------------------------------------|
| `name`    | `str`    | required   | Model name. Must match a registered model (see list below).              |
| `type`    | `str`    | `"tabular"`| Model category: `tabular`, `tabular_quantum`, or `imaging`.              |
| `preset`  | `str`    | `null`     | Per-model preset override: `quick`, `small`, `medium`, `large`, `exhaustive`. |
| `params`  | `dict`   | `{}`       | Explicit model parameters. Override preset values.                       |
| `enabled` | `bool`   | `true`     | Set to `false` to skip this model without removing it from the config.   |

### Registered model names

**Tabular (classical):** `xgboost`, `lightgbm`, `catboost`, `mlp`, `tabpfn`

**Tabular (quantum, angle-encoded):** `angle_pqk_svm` (uses `type: tabular`)

**Tabular (quantum, amplitude-encoded):** `kernel_svm`, `projected_kernel_svm`, `qcnn_muw`, `qcnn_alt`, `qdc_hadamard`, `quantum_gp`, `qnn`

**Imaging:** `cnn_2d`, `cnn_3d`

**Example:**
```yaml
models:
  - name: xgboost
    type: tabular
    preset: large
    params:
      n_estimators: 1000

  - name: kernel_svm
    type: tabular_quantum

  - name: qcnn_muw
    type: tabular_quantum
    preset: medium
    params:
      lr: 0.005

  - name: catboost
    type: tabular
    enabled: false  # skip this model
```

---

## evaluation

Evaluation and metrics configuration.

| Key                    | Type        | Default                                           | Description                                        |
|------------------------|-------------|---------------------------------------------------|----------------------------------------------------|
| `metrics`              | `list[str]` | `["balanced_accuracy", "auc", "sensitivity", "specificity"]` | Metrics to compute. See supported list below. |
| `primary_metric`       | `str`       | `"balanced_accuracy"`                             | Metric used for model comparison and ranking.      |
| `geometric_difference` | `bool`      | `false`                                           | Run Huang et al. 2021 geometric difference analysis.|
| `figure_dpi`           | `int`       | `300`                                             | Resolution for saved figures (dots per inch).      |

### Supported metrics

| Metric               | Key                   | Description                                |
|----------------------|-----------------------|--------------------------------------------|
| Balanced Accuracy    | `balanced_accuracy`   | Mean of per-class recall values.           |
| AUC                  | `auc`                 | Area under the ROC curve.                  |
| Sensitivity          | `sensitivity`         | True positive rate (recall for positive class). |
| Specificity          | `specificity`         | True negative rate.                        |
| Accuracy             | `accuracy`            | Overall classification accuracy.           |
| F1 Score             | `f1`                  | Harmonic mean of precision and recall.     |
| Precision            | `precision`           | Positive predictive value.                 |

**Example:**
```yaml
evaluation:
  metrics: [balanced_accuracy, auc, sensitivity, specificity]
  primary_metric: balanced_accuracy
  geometric_difference: true
  figure_dpi: 300
```

---

## explainability

Model explanation configuration.

| Key         | Type        | Default         | Description                                                    |
|-------------|-------------|-----------------|----------------------------------------------------------------|
| `methods`   | `list[str]` | `[]`            | Explanation methods to run: `shap`, `lime`, `gradcam`.         |
| `shap`      | `object`    | `null`          | SHAP-specific settings (see sub-keys below).                   |
| `lime`      | `object`    | `null`          | LIME-specific settings (see sub-keys below).                   |
| `gradcam`   | `object`    | `null`          | GradCAM-specific settings (see sub-keys below).                |

### explainability.shap

| Key               | Type   | Default  | Description                                                       |
|-------------------|--------|----------|-------------------------------------------------------------------|
| `n_samples`       | `int`  | `100`    | Number of background samples for SHAP KernelExplainer.            |
| `max_display`     | `int`  | `20`     | Maximum features to show in SHAP plots.                           |
| `plot_waterfall`  | `int`  | `5`      | Number of individual sample waterfall plots to generate.          |

### explainability.lime

| Key               | Type   | Default  | Description                                                       |
|-------------------|--------|----------|-------------------------------------------------------------------|
| `n_samples`       | `int`  | `5000`   | Number of perturbation samples for LIME.                          |
| `n_explanations`  | `int`  | `5`      | Number of individual sample explanations to generate.             |

### explainability.gradcam

| Key               | Type   | Default  | Description                                                       |
|-------------------|--------|----------|-------------------------------------------------------------------|
| `target_layer`    | `str`  | `null`   | Name of the CNN layer to visualize. If null, uses last conv layer.|

**Example:**
```yaml
explainability:
  methods: [shap, lime]
  shap:
    n_samples: 100
    max_display: 15
    plot_waterfall: 3
  lime:
    n_samples: 5000
    n_explanations: 3
```

---

## reporting

Report generation configuration.

| Key              | Type        | Default                        | Description                                          |
|------------------|-------------|--------------------------------|------------------------------------------------------|
| `formats`        | `list[str]` | `["latex", "markdown"]`        | Report formats to generate.                          |
| `latex`          | `object`    | `null`                         | LaTeX-specific settings (see sub-keys below).        |
| `include_plots`  | `bool`      | `true`                         | Include SHAP/LIME/GradCAM plots in reports.          |
| `include_methods`| `bool`      | `true`                         | Generate methods section describing analysis steps.  |
| `include_results`| `bool`      | `true`                         | Generate results section with metrics tables.        |

### reporting.latex

| Key              | Type     | Default             | Description                                             |
|------------------|----------|---------------------|---------------------------------------------------------|
| `methods_file`   | `str`    | `"methods.tex"`     | Output filename for methods section.                    |
| `results_file`   | `str`    | `"results.tex"`     | Output filename for results section.                    |
| `bib_file`       | `str`    | `"references.bib"`  | Output filename for BibTeX references.                  |
| `standalone`     | `bool`   | `false`             | Generate a complete compilable LaTeX document.          |

**Example:**
```yaml
reporting:
  formats: [latex, markdown]
  include_plots: true
  include_methods: true
  include_results: true
  latex:
    standalone: false
```

---

## Complete Configuration Example

```yaml
experiment:
  name: prostate_psma
  seed: 42
  results_dir: Results
  complexity: medium
  max_runtime_minutes: 120

data:
  type: tabular
  path: data/prostate_features.csv
  target: isup_grade
  separator: ";"
  id_column: patient_id

preprocessing:
  z_score: true
  feature_selection:
    method: mrmr
    max_features: 16
  binary_grouping:
    enabled: true
    positive_classes: [3, 4, 5]

cv:
  strategy: stratified_kfold
  n_folds: 5
  seeds: [42, 123, 456]

models:
  - name: xgboost
    type: tabular
  - name: lightgbm
    type: tabular
  - name: catboost
    type: tabular
  - name: kernel_svm
    type: tabular_quantum
  - name: qcnn_muw
    type: tabular_quantum
    params:
      lr: 0.005

evaluation:
  metrics: [balanced_accuracy, auc, sensitivity, specificity]
  primary_metric: balanced_accuracy
  geometric_difference: true
  figure_dpi: 300

explainability:
  methods: [shap, lime]
  shap:
    n_samples: 100
    max_display: 15

reporting:
  formats: [latex, markdown]
  include_plots: true
  include_methods: true
  include_results: true
```
