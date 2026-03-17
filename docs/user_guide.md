# CLARYON User Guide

A step-by-step tutorial for running your first CLARYON experiment.

---

## 1. Installation

### Basic installation

```bash
pip install claryon
```

### Installation from source

```bash
git clone https://github.com/laszlo-claryon/claryon.git
cd claryon
pip install -e .
```

### Optional dependencies

CLARYON has optional dependency groups for different features:

```bash
# Quantum models (PennyLane, Qiskit)
pip install claryon[quantum]

# Imaging models (PyTorch, torchvision, nibabel)
pip install claryon[imaging]

# Explainability (SHAP, LIME)
pip install claryon[explain]

# All optional dependencies
pip install claryon[all]
```

### Verify installation

```bash
claryon --version
claryon list-models
```

The `list-models` command prints all registered models and indicates which ones are available based on your installed dependencies.

---

## 2. Prepare Your Data

### Tabular data (CSV)

CLARYON expects a CSV file with one row per sample and columns for features and the target variable.

**Requirements:**
- Semicolon-separated by default (configurable via `data.separator`)
- First row must be column headers
- One column must contain the classification target (binary or multi-class)
- Optionally, one column for patient/sample IDs
- All feature columns must be numeric
- Missing values are not supported --- impute before using CLARYON

**Example CSV:**
```
patient_id;suv_max;volume;age;grade
P001;12.3;45.6;67;1
P002;8.1;23.4;54;0
P003;15.7;78.9;71;1
```

### Imaging data (NIfTI)

For imaging experiments, organize your data as follows:

```
data/
  nifti/
    patient_001.nii.gz
    patient_002.nii.gz
    ...
  labels.csv
```

The labels CSV maps filenames to class labels:
```
filename;grade
patient_001.nii.gz;1
patient_002.nii.gz;0
```

Both 2D (single-slice) and 3D NIfTI volumes are supported.

---

## 3. Write a Configuration File

Create a YAML file that describes your experiment. Here is an annotated example:

```yaml
# --- Experiment metadata ---
experiment:
  name: my_first_experiment   # Output subdirectory name
  seed: 42                     # Global random seed
  results_dir: Results         # Root output directory
  complexity: medium           # Preset level for all models

# --- Data source ---
data:
  type: tabular
  path: data/patients.csv      # Path to your CSV
  target: diagnosis            # Column name of the target variable
  separator: ";"               # CSV delimiter
  id_column: patient_id        # Optional: exclude this column from features

# --- Preprocessing ---
preprocessing:
  z_score: true                # Standardize features (recommended)
  feature_selection:
    method: mrmr               # Minimum Redundancy Maximum Relevance
    max_features: 16           # Keep at most 16 features

# --- Cross-validation ---
cv:
  strategy: stratified_kfold   # Preserves class balance in each fold
  n_folds: 5                   # Number of folds
  seeds: [42, 123]             # Run CV twice with different splits

# --- Models ---
models:
  - name: xgboost
    type: tabular

  - name: lightgbm
    type: tabular

  - name: kernel_svm
    type: tabular_quantum

# --- Evaluation ---
evaluation:
  metrics:
    - balanced_accuracy
    - auc
    - sensitivity
    - specificity
  primary_metric: balanced_accuracy

# --- Explainability ---
explainability:
  methods: [shap]
  shap:
    n_samples: 100

# --- Reporting ---
reporting:
  formats: [latex, markdown]
  include_methods: true
  include_results: true
```

Save this as `configs/my_experiment.yaml`.

### Validate your config before running

```bash
claryon validate-config -c configs/my_experiment.yaml
```

This checks for typos, missing required fields, and invalid model names without running anything.

---

## 4. Run an Experiment

```bash
claryon run -c configs/my_experiment.yaml
```

Add `-v` for stage-by-stage progress output:

```bash
claryon -v run -c configs/my_experiment.yaml
```

Add `-vv` for detailed per-fold logging:

```bash
claryon -vv run -c configs/my_experiment.yaml
```

### What happens during a run

1. **Loading data** --- reads your CSV or NIfTI files
2. **Binary grouping** --- converts multi-class targets to binary (if configured)
3. **Splitting** --- creates cross-validation folds
4. **Preprocessing** --- z-score normalization and feature selection (per fold, fitted on training data only)
5. **Training** --- trains each model on each fold and seed
6. **Evaluating** --- computes metrics on held-out test folds
7. **Explaining** --- generates SHAP/LIME explanations (if configured)
8. **Reporting** --- writes LaTeX and Markdown reports

At the end, CLARYON prints a summary table showing the mean performance of each model across folds and seeds.

---

## 5. Read the Results

After a run, your results directory looks like this:

```
Results/my_first_experiment/
  config_used.yaml                  # Copy of the config that was run
  run_info.json                     # Provenance metadata
  metrics_summary.csv               # Aggregated metrics for all models
  methods.tex                       # LaTeX methods section
  results.tex                       # LaTeX results section
  references.bib                    # BibTeX references
  report.md                         # Markdown summary report
  xgboost/
    seed_42/
      fold_0/
        Predictions.csv             # Per-sample predictions
        preprocessing_state.json    # Saved preprocessing parameters
        model_params.json           # Resolved model parameters
        model.joblib                # Saved model (classical)
      fold_1/
        ...
    seed_123/
      ...
    explanations/
      shap_summary_beeswarm.png     # SHAP feature importance
      shap_bar.png                  # Mean SHAP values
  kernel_svm/
    ...
  geometric_difference/             # If enabled
    geometric_difference_report.png
    gdq_results.json
```

### Key output files

**metrics_summary.csv** --- The main results file. Contains mean and standard deviation of each metric for each model across all folds and seeds.

**Predictions.csv** --- Per-fold predictions with columns for true label, predicted label, and prediction probability. Semicolon-separated.

**methods.tex** --- Ready-to-paste LaTeX text describing your preprocessing, models, and evaluation methodology. Includes citations.

**results.tex** --- LaTeX table of results, suitable for insertion into a manuscript.

**SHAP plots** --- Visual explanations of feature importance. The beeswarm plot shows both importance and direction of effect for each feature.

---

## 6. Run Inference on New Patients

After training, you can apply a saved model to new data without retraining:

```bash
claryon infer \
  --model-dir Results/my_first_experiment/xgboost/seed_42/fold_0/ \
  --input data/new_patients.csv \
  --output predictions_new.csv
```

This command:
1. Loads the saved preprocessing state (z-score parameters, feature selection mask)
2. Loads the saved model
3. Reads the new data from `--input`
4. Applies the same preprocessing that was used during training
5. Generates predictions
6. Writes results to `--output`

No configuration file is needed --- everything is stored in the model directory.

### Requirements for inference data

- Must have the same feature columns as the training data (before feature selection)
- Must use the same CSV separator
- The target column is not required (predictions only)
- The ID column, if present, will be included in the output

---

## 7. Troubleshooting

### "Model X not found in registry"

The model name in your config does not match any registered model. Run `claryon list-models` to see all available model names. Check for typos and note that names are case-sensitive.

### "No module named 'pennylane'" (or similar)

Quantum models require optional dependencies. Install them with:
```bash
pip install claryon[quantum]
```

Similarly, imaging models need `pip install claryon[imaging]` and explainability needs `pip install claryon[explain]`.

### "MEMORY WARNING: ... exceeds 80% of available RAM"

CLARYON detected that a model would likely exhaust your system memory. This usually happens with quantum models when the feature count is too high. Solutions:
- Reduce `max_features` in your feature selection config
- Use `qdc_hadamard` instead of `qdc_swap` (fewer qubits)
- Remove the offending model from your config

### "SKIPPING model_name: estimated memory exceeds available"

CLARYON automatically skipped a model to prevent a crash. The remaining models will continue running. Check the log for the specific memory estimate and consider reducing features.

### "OUT OF MEMORY during training"

Despite pre-flight checks, a model exceeded available memory during training. CLARYON caught the error and continued with the remaining models. Results for the failed model will show `status: oom` in the output.

### Predictions.csv is empty or missing

- Check that the model trained successfully (look for error messages in the log)
- Verify your data has no missing values
- Ensure feature columns are numeric

### SHAP/LIME plots not generated

- Verify `explainability.methods` includes `shap` or `lime` in your config
- Check that SHAP/LIME packages are installed: `pip install claryon[explain]`
- Some quantum models may not support SHAP/LIME --- check the log for warnings

### Config validation fails

Run `claryon validate-config -c your_config.yaml` to see specific validation errors. Common issues:
- Missing required fields (`data.path`, `data.target`)
- Invalid model names (check spelling)
- Invalid preset name (must be one of: quick, small, medium, large, exhaustive)
- YAML syntax errors (check indentation)

### Results differ between runs

- Ensure `experiment.seed` and `cv.seeds` are set explicitly
- Quantum models may have non-deterministic behavior on some platforms due to floating-point ordering
- CatBoost may use different thread counts across runs --- set `thread_count` in params for exact reproducibility

### Pipeline runs but all metrics are poor

- Check that your target column is correct and has the expected class distribution
- Verify that binary grouping (if enabled) groups the right classes as positive
- Try running with `-vv` to see per-fold details
- Check if feature selection is removing informative features (try without it first)
