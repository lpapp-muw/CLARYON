# CLAUDE.md — Preprocessing Pipeline + Infrastructure

## Project

**CLARYON** — CLassical-quantum AI for Reproducible Explainable OpeN-source medicine.
Quantum methods port is done (or running). This session builds the real preprocessing pipeline, mRMR feature selection, z-score normalization, binary grouping, BibTeX generation, and model-level method descriptions.

## READ FIRST

1. **WORKLOG.md** — Current state, hard facts, task list
2. **claryon/pipeline.py** — Current pipeline (stages 1-7)
3. **claryon/preprocessing/tabular_prep.py** — Existing preprocessing module
4. **claryon/config_schema.py** — Config structure

## ============================================================
## ARCHITECTURAL OVERVIEW — What changes
## ============================================================
##
## BEFORE this session:
##   load_data → passthrough_preprocess → split → train → evaluate → explain → report
##   (preprocessing was a no-op; models received raw features)
##
## AFTER this session:
##   load_data → binary_grouping → split → per_fold_preprocess → train → evaluate → explain → report
##                                         ↓
##                              z-score (fit on train, store coefficients)
##                              mRMR feature selection (fit on train, store mask)
##                              store PreprocessingState per fold
##                              apply to test fold using train coefficients
##
## For imaging (CNN/qCNN):
##   load_data → split → per_fold_image_normalize → train → ...
##                        ↓
##                 per-image OR cohort-global min-max (user-controlled)
##                 store normalization params per fold
##
## Key new dataclass:
##   PreprocessingState(feature_mask, z_mean, z_std, image_norm_min, image_norm_max)
##   Saved as preprocessing_state.json per model/seed/fold
##   Loaded at inference time before prediction
##
## ============================================================

## ============================================================
## TASKS (execute in order)
## ============================================================

### Task 1: Define PreprocessingState dataclass

**File**: `claryon/preprocessing/state.py` (NEW)

```python
@dataclass
class PreprocessingState:
    """Stores all preprocessing parameters fitted on training data.
    
    Saved per model/seed/fold. Loaded before inference.
    """
    # Z-score normalization
    z_mean: np.ndarray          # shape (n_features_original,)
    z_std: np.ndarray           # shape (n_features_original,)
    
    # mRMR feature selection
    selected_features: List[int]  # indices into original feature space
    selected_feature_names: List[str]  # original column names
    spearman_threshold: float
    
    # Image normalization (for CNN/qCNN)
    image_norm_mode: str        # "per_image" or "cohort_global"
    image_norm_min: Optional[float]   # cohort global min (from train)
    image_norm_max: Optional[float]   # cohort global max (from train)
    
    # Metadata
    n_features_original: int
    n_features_selected: int
    
    def save(self, path: Path) -> None: ...
    
    @staticmethod
    def load(path: Path) -> PreprocessingState: ...
    
    def apply_tabular(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalize + select features. Uses stored train coefficients."""
        X_z = (X - self.z_mean) / np.maximum(self.z_std, 1e-12)
        return X_z[:, self.selected_features]
    
    def apply_image(self, volumes: np.ndarray) -> np.ndarray:
        """Normalize image volumes using stored parameters."""
        ...
```

**Test**: `tests/test_preprocessing/test_state.py` — round-trip save/load, apply consistency

### Task 2: Implement mRMR feature selection

**File**: `claryon/preprocessing/feature_selection.py` (NEW)

**Algorithm**:
1. Compute Spearman correlation matrix on training features
2. Build redundancy clusters: features with |ρ| > threshold are redundant
3. Within each cluster, pick the feature with highest |Spearman ρ| to the target label
4. Return selected feature indices

```python
def mrmr_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    spearman_threshold: float = 0.8,
    max_features: Optional[int] = None,
) -> Tuple[List[int], List[str]]:
    """Minimum Redundancy Maximum Relevance feature selection.
    
    Args:
        X_train: Training features (n_samples, n_features).
        y_train: Training labels (n_samples,).
        feature_names: Column names for logging.
        spearman_threshold: Features with |ρ| > threshold are redundant. Default 0.8.
        max_features: Optional hard cap on output features.
    
    Returns:
        (selected_indices, selected_names)
    """
```

**Guard**: If n_features <= 4, skip mRMR and return all features.

**Test**: `tests/test_preprocessing/test_feature_selection.py`
- Test on synthetic data where 2 features are perfectly correlated
- Verify the right one (higher relevance to label) is kept
- Verify threshold=1.0 keeps everything
- Verify threshold=0.0 keeps only 1 feature per cluster

### Task 3: Implement z-score normalization as stateful preprocessor

**File**: Update `claryon/preprocessing/tabular_prep.py`

Add:
```python
def fit_zscore(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute z-score parameters from training data.
    Returns (mean, std) arrays."""

def apply_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score using pre-computed parameters."""
```

These are simple but must be explicitly separated from fit/apply to prevent leakage.

**Test**: Verify that z-score params fitted on train are applied to test (NOT re-fitted on test).

### Task 4: Implement image normalization

**File**: Update `claryon/preprocessing/image_prep.py`

Two modes:
```python
def normalize_images(
    volumes: np.ndarray,
    mode: str = "per_image",       # "per_image" or "cohort_global"
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """Normalize image volumes to [0, 1].
    
    mode="per_image": each volume scaled independently.
    mode="cohort_global": use global_min/max from training set.
    
    Returns (normalized_volumes, used_min, used_max)
    """
```

**Test**: Verify per-image mode produces [0,1] range per volume. Verify cohort-global uses training set bounds.

### Task 5: Add binary grouping config + implementation

**File**: Update `claryon/config_schema.py`

Add to DataConfig or CVConfig:
```python
class BinaryGroupingConfig(BaseModel):
    """User-defined binary reduction of multiclass labels."""
    enabled: bool = False
    positive: List[Any] = Field(default_factory=list)  # label values → class 1
    negative: List[Any] = Field(default_factory=list)  # label values → class 0
    # If negative is empty, everything NOT in positive becomes negative
```

Add to root config:
```python
class ClaryonConfig(BaseModel):
    ...
    binary_grouping: Optional[BinaryGroupingConfig] = None
```

**Implementation** in pipeline: After loading data, if `binary_grouping.enabled`:
1. Map labels in `positive` → 1, everything else (or `negative` list) → 0
2. Update `dataset.task_type` to BINARY
3. Update `dataset.label_mapper` to BinaryLabelMapper
4. Log: "Binary grouping applied: positive={positive}, negative={negative}"

**Test**: 4-class data → group [2,3] as positive, [0,1] as negative → verify binary labels

### Task 6: Add preprocessing config to schema

**File**: Update `claryon/config_schema.py`

```python
class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""
    zscore: bool = True                          # z-score normalize features
    feature_selection: bool = True               # run mRMR
    spearman_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    max_features: Optional[int] = None           # hard cap after mRMR
    image_normalization: Literal["per_image", "cohort_global"] = "per_image"
```

Add to root config:
```python
class ClaryonConfig(BaseModel):
    ...
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
```

### Task 7: Rewire pipeline.py — the big change

**This is the critical task.** The preprocessing must happen INSIDE the fold loop, not before splitting.

Current flow:
```
stage_load_data → stage_preprocess(no-op) → stage_split → stage_train(per fold)
```

New flow:
```
stage_load_data → stage_binary_grouping → stage_split → stage_train_with_preprocess(per fold):
    for each seed/fold:
        1. Split into X_train, X_test
        2. Fit z-score on X_train → store μ, σ
        3. Apply z-score to X_train and X_test
        4. Fit mRMR on X_train → store feature mask
        5. Apply feature mask to X_train and X_test
        6. (For quantum models) Amplitude encode
        7. Train model
        8. Save PreprocessingState to disk
        9. Predict on X_test
        10. Write Predictions.csv
```

**For inference (external test set, no labels)**:
```
1. Load PreprocessingState from disk
2. Apply z-score using stored μ, σ
3. Apply feature mask using stored indices
4. (For quantum) Amplitude encode with stored pad_len
5. Predict
```

**Key constraint**: PreprocessingState is PER FOLD PER SEED. Each fold may select different features (because mRMR depends on the training subset). This is correct — it prevents information leakage.

**Storage layout**:
```
Results/<experiment>/
  <model>/seed_<s>/fold_<f>/
    Predictions.csv
    preprocessing_state.json      ← NEW
    model.joblib (if model supports save)
```

### Task 8: Generate BibTeX file

**File**: `claryon/reporting/references.bib` (NEW)

Generate from all cite keys in `method_descriptions.yaml`. Include:

```bibtex
@article{Chen2016,
  author = {Chen, Tianqi and Guestrin, Carlos},
  title = {{XGBoost}: A Scalable Tree Boosting System},
  journal = {Proceedings of the 22nd ACM SIGKDD},
  year = {2016},
  doi = {10.1145/2939672.2939785}
}

@article{Ke2017, ... }  % LightGBM
@article{Prokhorenkova2018, ... }  % CatBoost
@article{Havlicek2019, ... }  % Quantum kernel
@article{Moradi2022, ... }  % Clinical data classification
@article{Moradi2023, ... }  % Error mitigation PET radiomics
@article{PappQCNN, note={Under revision}, ... }  % QCNN MUW
@article{Lundberg2017, ... }  % SHAP
@article{Ribeiro2016, ... }  % LIME
@article{Demsar2006, ... }  % Friedman/Nemenyi
@article{Schuld2018, ... }  % Quantum encoding
@article{Hollmann2023, ... }  % TabPFN
@article{Selvaraju2017, ... }  % GradCAM
@software{claryon2026, ... }  % Self-citation
```

Also add a function to auto-generate the .bib from the YAML:
```python
def generate_bibtex(descriptions_path, output_path):
    """Scan method_descriptions.yaml for cite keys, 
    write references.bib with all needed entries."""
```

### Task 9: Hybrid method description fallback

**File**: Update `claryon/models/base.py` and `claryon/reporting/structured_report.py`

Add optional class attribute to ModelBuilder:
```python
class ModelBuilder(ABC):
    ...
    # Optional: model provides its own method description
    method_description: ClassVar[str] = ""
    method_cite_key: ClassVar[str] = ""
```

Update `structured_report.py` `_section_models()`:
```python
# Priority: YAML registry → model class attribute → generic fallback
text = _get_text(registry, "models", name, ctx)
if not text:
    model_cls = get("model", name)
    if hasattr(model_cls, "method_description") and model_cls.method_description:
        text = model_cls.method_description
        # Interpolate params
if not text:
    text = f"The \\textbf{{{name}}} model was included in the comparison."
```

### Task 10: Remove DICOM from config schema

**File**: `claryon/config_schema.py`

Change:
```python
format: Literal["nifti", "tiff", "dicom"] = "nifti"
```
To:
```python
format: Literal["nifti", "tiff"] = "nifti"
```

### Task 11: Ensemble reporting in methods + results .tex

**File**: Update `claryon/reporting/structured_report.py` and `claryon/reporting/latex_report.py`

If ensemble is configured:
- Methods.tex: "An ensemble prediction was computed by softmax averaging of the per-class probability vectors from {model_list}."
- Results.tex: Add an "Ensemble" row in the metrics table alongside individual models

### Task 12: Update param_descriptions in method_descriptions.yaml

Add missing parameter descriptions for new quantum methods:
```yaml
param_descriptions:
  noise: "a noise parameter of {{value}} was used for kernel regularization"
  margin: "a margin of {{value}} was used in the multi-class hinge loss"
  num_layers: "the circuit depth was {{value}} variational layers"
  spearman_threshold: "redundant features with Spearman ρ > {{value}} were clustered"
  image_normalization: "image volumes were normalized using {{value}} strategy"
```

### Task 13: Update example configs

Create `configs/iris_full_preprocess.yaml` that exercises the full pipeline:
```yaml
experiment:
  name: iris_full_preprocess
  seed: 42
  results_dir: Results/iris_preprocess

data:
  tabular:
    path: demo_data/iris/iris.csv
    label_col: label
    id_col: Key
    sep: ";"

binary_grouping:
  enabled: true
  positive: [1, 2]    # versicolor + virginica
  negative: [0]        # setosa

preprocessing:
  zscore: true
  feature_selection: true
  spearman_threshold: 0.8
  image_normalization: per_image

cv:
  strategy: kfold
  n_folds: 5
  seeds: [42]

models:
  - name: xgboost
    type: tabular
  - name: kernel_svm
    type: tabular_quantum
  - name: qcnn_muw
    type: tabular_quantum
    params:
      epochs: 10

explainability:
  shap: true
  lime: true

evaluation:
  metrics: [bacc, auc, sensitivity, specificity]

reporting:
  markdown: true
  latex: true
```

### Task 14: Update .gitignore

Add:
```
catboost_info/
*.egg-info/
fix_log.txt
```

## ============================================================
## WORKFLOW
## ============================================================
##
## For each task:
##   1. Write/update module + tests
##   2. Run: python -m pytest tests/ -x -q --timeout=300
##   3. Commit: git add -A && git commit -m "preprocess: <description>"
##
## After all tasks:
##   1. Run full test suite: python -m pytest tests/ -q --timeout=600
##   2. Run the full preprocessed experiment:
##      python -m claryon -v run -c configs/iris_full_preprocess.yaml
##   3. Verify:
##      - preprocessing_state.json exists per fold
##      - Selected features < original features (mRMR worked)
##      - Predictions.csv exists for all models
##      - metrics_summary.csv shows results
##      - methods.tex describes preprocessing steps
##      - results.tex shows metrics table
##   4. git tag v0.10.0-preprocess
##   5. Update WORKLOG.md
##
## ============================================================

## Code rules (same as always)

- `from __future__ import annotations`
- Type hints, Google docstrings, logging not print
- `@register` decorators where applicable
- Deterministic seeding
- All predictions through io/predictions.py
- CSV separator: `;`

## Fix loop protocol

Same as before. 5 attempts max per error, then BLOCKED.

## Stop conditions

- Need to change ModelBuilder.fit() signature (ask first)
- mRMR implementation unclear
- PreprocessingState serialization format unclear
- Pipeline flow change breaks existing tests

## Key hard facts

- HF-004: Quantum models cluster near 0.5 → Youden's J threshold critical
- HF-006: 306 features → 512 pad → 9 qubits. mRMR should reduce this substantially.
- HF-010: n_qubits auto-derived from encoding. mRMR output feeds into this chain.
- Z-score MUST be fitted on train only. Feature selection MUST be fitted on train only. This is non-negotiable for scientific validity.
- PreprocessingState is PER FOLD PER SEED. Different folds may select different features.
- Guard: if n_features <= 4, skip mRMR.
