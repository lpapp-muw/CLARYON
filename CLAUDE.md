# CLAUDE.md — Final Polish: Presets, Auto Mode, GDQ, CLI, Docs, Cleanup

## Project

**CLARYON** — CLassical-quantum AI for Reproducible Explainable OpeN-source medicine.
Preprocessing pipeline complete (v0.10.0). This session adds model presets, auto complexity mode, resource safety, full Geometric Difference framework, model save/load + inference, SHAP/LIME plots, professional CLI output, documentation, and cleanup.

## READ FIRST

1. **WORKLOG.md** — Current state
2. **claryon/pipeline.py** — Pipeline orchestrator
3. **claryon/config_schema.py** — Config structure
4. **claryon/models/base.py** — ModelBuilder ABC

## ============================================================
## TASK DEPENDENCY GRAPH
## ============================================================
##
##  Group A (foundation — do first):
##    T1  presets.yaml + preset resolution
##    T2  auto mode (depends on T1)
##    T3  resource estimator + OOM protection (depends on T2)
##
##  Group B (model persistence — do second):
##    T4  model save/load wiring in pipeline
##    T5  claryon infer subcommand (depends on T4)
##    T6  provenance metadata (run_info.json, config copy)
##
##  Group C (output quality — do third):
##    T7  CLI progress formatting + summary table
##    T8  CLI ASCII banner
##    T9  SHAP/LIME plot generation
##    T10 full Geometric Difference framework + visualization
##
##  Group D (documentation — do fourth):
##    T11 docs/model_guide.md (presets, auto mode, parameter explanations)
##    T12 docs/config_reference.md (auto-generated from Pydantic schema)
##    T13 docs/user_guide.md
##    T14 docs/architecture.md
##    T15 docs/contributor_guide.md
##
##  Group E (cleanup — do last):
##    T16 remove Claude session files from repo
##    T17 remove source_archive/ (after confirming all ports)
##    T18 cleanup build artifacts, update .gitignore
##    T19 update README.md with all new features
##    T20 final integration test on all demo configs
##
## ============================================================

## ============================================================
## GROUP A: PRESETS + AUTO MODE + SAFETY
## ============================================================

### Task 1: Model presets system

**Files**:
- `claryon/models/presets.yaml` (NEW) — preset definitions per model and category
- `claryon/models/preset_resolver.py` (NEW) — loads presets, resolves for a ModelEntry
- Update `claryon/config_schema.py` — add `preset` field to ModelEntry, `complexity` to ExperimentConfig
- Update `claryon/pipeline.py` — resolve presets before model construction

**Presets YAML structure**:
```yaml
# Category defaults (inherited by models in that category)
_defaults:
  tabular:
    quick:   { n_estimators: 50, max_depth: 4, learning_rate: 0.1 }
    small:   { n_estimators: 200, max_depth: 6, learning_rate: 0.05 }
    medium:  { n_estimators: 500, max_depth: 8, learning_rate: 0.02 }
    large:   { n_estimators: 1000, max_depth: 10, learning_rate: 0.01 }
    exhaustive: { n_estimators: 2000, max_depth: 12, learning_rate: 0.005 }

  tabular_quantum:
    quick:   { epochs: 5, lr: 0.05, batch_size: 32 }
    small:   { epochs: 30, lr: 0.02, batch_size: 16 }
    medium:  { epochs: 100, lr: 0.01, batch_size: 16 }
    large:   { epochs: 300, lr: 0.005, batch_size: 8 }
    exhaustive: { epochs: 500, lr: 0.002, batch_size: 8 }

  imaging:
    quick:   { epochs: 5, batch_size: 8 }
    small:   { epochs: 20, batch_size: 8 }
    medium:  { epochs: 50, batch_size: 4 }
    large:   { epochs: 100, batch_size: 4 }
    exhaustive: { epochs: 200, batch_size: 2 }

# Per-model overrides (merged on top of category defaults)
xgboost:
  medium: { n_estimators: 500 }

catboost:
  medium: { iterations: 500, verbose: 0 }

kernel_svm:
  # No training loop — presets affect shots only
  quick:   { shots: null }
  small:   { shots: null }
  medium:  { shots: null }
  large:   { shots: 8192 }

qcnn_muw:
  medium:  { epochs: 100, lr: 0.01, init_scale: 0.1 }

qnn:
  medium:  { num_layers: 4, epochs: 100, lr: 0.005, margin: 0.15 }
```

**Resolution priority** (highest wins):
1. Explicit `params` in YAML config
2. Model-level `preset`
3. Global `complexity` setting
4. Category default for `medium`

**Config schema additions**:
```python
class ExperimentConfig(BaseModel):
    name: str = "experiment"
    seed: int = 42
    results_dir: str = "Results"
    complexity: Literal["quick", "small", "medium", "large", "exhaustive", "auto"] = "medium"
    max_runtime_minutes: int = Field(default=120, ge=1)

class ModelEntry(BaseModel):
    name: str
    type: Literal["tabular", "tabular_quantum", "imaging"] = "tabular"
    preset: Optional[Literal["quick", "small", "medium", "large", "exhaustive"]] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
```

**Runtime warning**: If a quantum model resolves to `quick` or `small`:
```
WARNING: qcnn_muw using 'quick' preset (5 epochs). For publishable results, use 'medium' or higher.
```

**Test**: `tests/test_models/test_presets.py`
- Verify resolution priority (explicit params override preset)
- Verify category fallback works
- Verify warning is logged for quick/small quantum presets

### Task 2: Auto complexity mode

**File**: `claryon/models/auto_complexity.py` (NEW)

**Algorithm**:
```python
def auto_select_presets(
    config: ClaryonConfig,
    n_samples: int,
    n_features: int,
    n_features_after_mrmr: int,
) -> Dict[str, str]:
    """Select preset per model based on dataset + time budget.
    
    Returns dict mapping model_name → preset_name.
    """
    n_qubits = ceil(log2(max(n_features_after_mrmr, 2)))
    budget_seconds = config.experiment.max_runtime_minutes * 60
    n_folds = config.cv.n_folds if config.cv.strategy != "holdout" else 1
    n_seeds = len(config.cv.seeds)
    n_models = len(config.models)
    
    budget_per_model_fold = budget_seconds / max(n_models * n_folds * n_seeds, 1)
    
    selected = {}
    for model_entry in config.models:
        estimates = {}
        for preset_name in ["exhaustive", "large", "medium", "small", "quick"]:
            params = resolve_preset(model_entry, preset_name)
            est = estimate_runtime(model_entry.name, model_entry.type,
                                   n_samples, n_qubits, params)
            estimates[preset_name] = est
        
        # Pick highest quality that fits budget
        chosen = "quick"
        for preset_name in ["exhaustive", "large", "medium", "small", "quick"]:
            if estimates[preset_name] <= budget_per_model_fold:
                chosen = preset_name
                break
        
        selected[model_entry.name] = chosen
    
    return selected
```

**Runtime estimator** (rough but useful):
```python
def estimate_runtime(model_name, model_type, n_samples, n_qubits, params):
    """Estimate seconds per fold."""
    if model_type == "tabular":
        return 5.0  # classical models are essentially instant
    
    circuit_cost = (2 ** n_qubits) * 1e-4  # seconds per circuit eval (simulator)
    
    if model_name == "kernel_svm":
        return n_samples ** 2 * circuit_cost  # N² kernel matrix
    elif model_name in ("qdc_hadamard", "qdc_swap"):
        return n_samples ** 2 * circuit_cost
    elif model_name == "qdc_swap":
        # SWAP test uses 2n+1 qubits
        swap_cost = (2 ** (2 * n_qubits + 1)) * 1e-4
        return n_samples ** 2 * swap_cost
    else:
        # Training-based quantum models
        epochs = params.get("epochs", 100)
        batch_size = params.get("batch_size", 16)
        batches = max(n_samples // batch_size, 1)
        return epochs * batches * circuit_cost * 3  # 3x for gradients
```

**Pipeline integration**: After mRMR (when n_features_after_mrmr is known), if `complexity == "auto"`:
1. Call `auto_select_presets()`
2. Log each decision with estimated runtime
3. Save resolved config to `Results/<experiment>/auto_resolved_config.yaml`

**Test**: `tests/test_models/test_auto_complexity.py`
- Verify auto selects lower preset for high qubit counts
- Verify budget constraint is respected
- Verify classical models always get at least medium

### Task 3: Resource estimator + OOM protection

**File**: `claryon/safety.py` (NEW)

**Pre-flight checks before training**:
```python
def preflight_resource_check(
    model_name: str,
    model_type: str,
    n_samples: int,
    n_qubits: int,
    params: Dict[str, Any],
) -> List[str]:
    """Check for potential resource issues. Returns list of warnings."""
    warnings = []
    
    # Memory: quantum state vector = 2^n_qubits complex numbers = 16 * 2^n bytes
    state_vector_bytes = 16 * (2 ** n_qubits)
    if state_vector_bytes > 1e9:  # > 1 GB
        warnings.append(
            f"MEMORY WARNING: {model_name} requires {state_vector_bytes/1e9:.1f} GB "
            f"for state vector alone ({n_qubits} qubits). "
            f"Consider reducing features via mRMR or max_features."
        )
    
    # Memory: kernel matrix for kernel-based models
    if model_name in ("kernel_svm", "sq_kernel_svm", "qdc_hadamard", "qdc_swap", "quantum_gp"):
        kernel_bytes = n_samples ** 2 * 8  # float64
        if kernel_bytes > 2e9:  # > 2 GB
            warnings.append(
                f"MEMORY WARNING: {model_name} kernel matrix needs "
                f"{kernel_bytes/1e9:.1f} GB ({n_samples}² entries). "
                f"Consider subsampling or using a training-based quantum model."
            )
    
    # SWAP test: 2n+1 qubits
    if model_name == "qdc_swap":
        swap_qubits = 2 * n_qubits + 1
        swap_bytes = 16 * (2 ** swap_qubits)
        if swap_bytes > 1e9:
            warnings.append(
                f"MEMORY WARNING: qdc_swap uses {swap_qubits} qubits "
                f"({swap_bytes/1e9:.1f} GB state vector). "
                f"Consider qdc_hadamard ({n_qubits+1} qubits) instead."
            )
    
    # Qubit count warning
    if n_qubits > 20:
        warnings.append(
            f"RUNTIME WARNING: {n_qubits} qubits — simulation cost O(2^{n_qubits}). "
            f"Estimated memory: {state_vector_bytes/1e9:.1f} GB. "
            f"This WILL be extremely slow. Reduce features."
        )
    elif n_qubits > 15:
        warnings.append(
            f"RUNTIME WARNING: {n_qubits} qubits — expect long runtimes. "
            f"Consider setting max_features to limit qubit count."
        )
    
    # Runtime estimate
    estimated_seconds = estimate_runtime(model_name, model_type, n_samples, n_qubits, params)
    if estimated_seconds > 3600:
        warnings.append(
            f"RUNTIME WARNING: {model_name} estimated at {estimated_seconds/3600:.1f} hours per fold."
        )
    
    return warnings
```

**Pipeline integration**: Before each model trains, run preflight check. Log all warnings. If any MEMORY WARNING with >90% of available RAM (check via `psutil.virtual_memory()` or `/proc/meminfo`), SKIP the model with an error message instead of crashing:

```python
available_gb = get_available_memory_gb()
estimated_gb = estimate_memory_usage(model_name, n_qubits, n_samples)
if estimated_gb > 0.8 * available_gb:
    logger.error(
        "SKIPPING %s: estimated memory %.1f GB exceeds 80%% of available %.1f GB. "
        "Reduce features or use a different model.",
        model_name, estimated_gb, available_gb
    )
    continue  # skip this model, don't crash
```

**Graceful degradation**, never crash. If a model does crash with MemoryError despite checks, catch it:
```python
try:
    model.fit(X_train, y_train, task_type)
except MemoryError:
    logger.error("OUT OF MEMORY during %s training. Skipping.", model_name)
    model_results.append({"seed": seed, "fold": fold, "status": "oom"})
    continue
except Exception as e:
    logger.error("FAILED %s: %s", model_name, e)
    model_results.append({"seed": seed, "fold": fold, "status": "error", "error": str(e)})
    continue
```

**Test**: `tests/test_safety.py`
- Verify warnings generated for 20+ qubits
- Verify warnings for large kernel matrices
- Verify no crash on MemoryError (mock test)

## ============================================================
## GROUP B: MODEL PERSISTENCE + INFERENCE
## ============================================================

### Task 4: Model save/load wiring in pipeline

**Changes to pipeline.py**:

After `model.fit()`, save the model:
```python
model_dir = pred_dir  # same as Predictions.csv directory
if hasattr(model, "save"):
    model.save(model_dir)
    logger.info("  Model saved to %s", model_dir)
```

Ensure all model classes implement `save()` and `load()`:
- Classical models: joblib dump/load (most already have this)
- Quantum models: save weights as numpy arrays + metadata JSON
- CNNs: save PyTorch state_dict

**Also save**: the resolved params (after preset resolution) as `model_params.json` in the fold directory.

**Test**: Verify save + load round-trip for at least 1 classical, 1 quantum, 1 CNN model.

### Task 5: `claryon infer` subcommand

**File**: Update `claryon/cli.py` + NEW `claryon/inference.py`

```bash
claryon infer \
    --model-dir Results/iris_full/xgboost/seed_42/fold_0/ \
    --input data/new_patients.csv \
    --output predictions_new.csv
```

**Logic**:
1. Load `preprocessing_state.json` from model-dir
2. Load saved model from model-dir
3. Load new data from --input
4. Apply preprocessing state (z-score with stored μ/σ, feature selection with stored mask)
5. (If quantum) Amplitude encode with stored pad_len
6. Predict
7. Write predictions to --output

**Config not needed** — everything is stored in the model directory.

**Test**: `tests/test_integration/test_inference.py`
- Train on iris, save model + state
- Load and predict on held-out data
- Verify predictions match direct predict

### Task 6: Provenance metadata

**File**: Update pipeline.py

After pipeline completes, write to results_dir:

```python
# run_info.json
{
    "claryon_version": "0.11.0",
    "python_version": "3.11.15",
    "timestamp": "2026-03-17T10:30:00Z",
    "hostname": "omega",
    "config_hash": "sha256:abc123...",
    "git_commit": "3514267",
    "runtime_seconds": 1234.5,
    "n_models": 7,
    "n_folds": 5,
    "n_seeds": 2
}
```

Also copy the YAML config into results_dir as `config_used.yaml`.

If `complexity: auto`, also save `auto_resolved_config.yaml` with the resolved parameters.

## ============================================================
## GROUP C: OUTPUT QUALITY
## ============================================================

### Task 7: CLI progress formatting + summary table

**File**: `claryon/progress.py` (NEW) + update `pipeline.py`

**Stage progress**:
```
[1/8] Loading data...                          ✓ 150 samples × 4 features
[2/8] Binary grouping...                       ✓ 3-class → binary (positive: [1, 2])
[3/8] Splitting...                             ✓ 5 folds × 2 seeds = 10 splits
[4/8] Training...
      xgboost     ████████████████████ 10/10 folds  [00:05]
      kernel_svm  ████████░░░░░░░░░░░░  4/10 folds  [02:31]
[5/8] Evaluating...                            ✓ metrics_summary.csv written
[6/8] Explaining...                            ✓ SHAP + LIME for 7 models
[7/8] Reporting...                             ✓ methods.tex + results.tex + report.md
[8/8] Done.
```

**Summary table at end** (using `tabulate` package, already installed):
```
╒══════════════╤════════╤════════╤═════════════╤═════════════╕
│ Model        │  BACC  │  AUC   │ Sensitivity │ Specificity │
╞══════════════╪════════╪════════╪═════════════╪═════════════╡
│ xgboost      │ 0.9950 │ 0.9950 │      0.9900 │      1.0000 │
│ kernel_svm   │ 1.0000 │ 1.0000 │      1.0000 │      1.0000 │
│ qcnn_muw     │ 1.0000 │ 1.0000 │      1.0000 │      1.0000 │
╘══════════════╧════════╧════════╧═════════════╧═════════════╛

Results saved to: Results/iris_full/
Runtime: 4m 32s
```

**Implementation**: Use `tabulate` for the table. Use simple print with `\r` for progress bars (no external dependency). All output goes to stderr (so stdout can be piped if needed). Logging remains on logging module. Progress display is separate.

**Respect verbosity**: `-v` shows stage progress + summary table. `-vv` shows stage progress + per-fold logs + summary. No flags: only summary table at end.

### Task 8: CLI ASCII banner

**File**: Update `claryon/cli.py`

Add a constant:
```python
BANNER = r"""
   _____ _        _    ______   _____  _   _
  / ____| |      / \  |  _ \ \ / / _ \| \ | |
 | |    | |     / _ \ | |_) \ V / | | |  \| |
 | |    | |    / ___ \|  _ < | || |_| | |\  |
  \____|_|___/_/   \_\_| \_\|_| \___/|_| \_|

  CLassical-quantum AI for Reproducible
  Explainable OpeN-source medicine       v{version}
"""
```

Print at startup before any logging. Only print for `run` and `infer` commands, not for `list-models` etc.

### Task 9: SHAP/LIME plot generation

**File**: Update `claryon/explainability/shap_.py` and `lime_.py`, or new `claryon/explainability/plots.py`

After computing SHAP values, generate and save:
- `shap_summary_beeswarm.png` — beeswarm plot (feature importance + direction)
- `shap_bar.png` — mean |SHAP value| per feature
- `shap_waterfall_sample_N.png` — per-sample waterfall (for top N samples)

After computing LIME explanations, generate and save:
- `lime_explanation_sample_N.png` — per-sample local explanation bar chart

All plots:
- matplotlib/seaborn, saved as PNG at config `figure_dpi` (default 300)
- Self-contained (no external display needed)
- Feature names on axes (from dataset.feature_names)
- Stored in `<model>/explanations/` directory alongside the raw data files

**Test**: Verify plot files are created. Verify they are valid PNG (header check).

### Task 10: Full Geometric Difference framework

**File**: Update `claryon/evaluation/geometric_difference.py`

Implement the full Huang et al. 2021 framework:

```python
def geometric_difference(K_C: np.ndarray, K_Q: np.ndarray) -> float:
    """g(K^C || K^Q) = sqrt(spectral_norm(K_Q^½ @ K_C^{-1} @ K_Q^½))"""
    # Already exists from Moradi port

def model_complexity(K: np.ndarray, y: np.ndarray) -> float:
    """s_K(N) = y^T @ K^{-1} @ y"""
    K_inv = np.linalg.pinv(K + 1e-8 * np.eye(len(K)))
    return float(y @ K_inv @ y)

def effective_dimension(K_Q: np.ndarray, threshold: float = 0.01) -> int:
    """d = rank(K_Q) with eigenvalue truncation."""
    eigenvalues = np.linalg.eigvalsh(K_Q)
    return int(np.sum(eigenvalues > threshold * eigenvalues.max()))

def quantum_advantage_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    quantum_kernel_fn: Callable,
    classical_kernels: Dict[str, Callable],
) -> Dict[str, Any]:
    """Run the full Huang et al. flowchart.
    
    Returns:
        {
            "g_CQ": {kernel_name: g_score},       # geometric difference per classical kernel
            "s_C": {kernel_name: s_score},          # model complexity per classical kernel
            "s_Q": float,                           # quantum model complexity
            "d": int,                               # effective dimension
            "K_Q_rank": int,                        # rank of quantum kernel
            "recommendation": str,                  # "quantum_advantage_likely" / "classical_sufficient" / "inconclusive"
            "explanation": str,                     # human-readable interpretation
        }
    """
```

**Decision logic** (from Huang et al. Figure 1):
```python
if all g_CQ values are small (~1):
    recommendation = "classical_sufficient"
    explanation = "Small geometric difference: classical ML is guaranteed competitive."
elif s_Q << s_C:
    recommendation = "quantum_advantage_likely"
    explanation = f"Large g_CQ ({max_g:.2f}) and lower quantum complexity suggest advantage."
else:
    recommendation = "inconclusive"
    explanation = "Large geometric difference but model complexities are similar."
```

**Classical kernels to test against**: linear, RBF (with gamma tuning), polynomial.

**Visualization**: Generate `geometric_difference_report.png`:
- Panel 1: g_CQ values as bar chart (per classical kernel)
- Panel 2: s_C vs s_Q scatter
- Panel 3: Eigenvalue spectrum of K_Q (showing effective dimension)
- Panel 4: Decision flowchart with computed values annotated

**Integration**: Optional in config:
```yaml
evaluation:
  geometric_difference: true   # run GDQ analysis after quantum models
```

When enabled, runs after training quantum models (needs their kernel matrices). Results go into `Results/<experiment>/geometric_difference/`.

**Methods.tex**: Add paragraph describing the GDQ analysis. Cite Huang2021.

**Results.tex**: Add GDQ table showing g_CQ, s_C, s_Q, d, and recommendation.

**BibTeX**: Add:
```bibtex
@article{Huang2021,
  author = {Huang, Hsin-Yuan and Broughton, Michael and Mohseni, Masoud and Babbush, Ryan and Boixo, Sergio and Neven, Hartmut and McClean, Jarrod R.},
  title = {Power of data in quantum machine learning},
  journal = {Nature Communications},
  volume = {12},
  pages = {2631},
  year = {2021},
  doi = {10.1038/s41467-021-22539-9}
}
```

**Test**: `tests/test_evaluation/test_geometric_difference.py`
- Test on synthetic data where quantum kernel is identical to classical → g ≈ 1
- Test on data where kernels differ → g > 1
- Test effective dimension on identity matrix → d = N

## ============================================================
## GROUP D: DOCUMENTATION
## ============================================================

### Task 11: docs/model_guide.md

**Comprehensive guide for non-quantum users.** Written for nuclear medicine researchers.

Sections:
1. **Choosing complexity** — quick/small/medium/large/exhaustive explained in plain English
2. **Auto mode** — what it does, when to trust it, when to override
3. **Understanding quantum models** — 2 paragraphs per model, no math, just "what it does and when it helps"
4. **Parameter reference** — table per model showing every parameter, its default, what it controls, and how it affects quality vs speed
5. **Runtime expectations** — table showing estimated runtime per model for different dataset sizes (100/500/1000 samples, 8/16/32 features)
6. **Common pitfalls** — "don't publish with quick preset", "reduce features before quantum", "kernel models scale as N²"
7. **When to use quantum** — honest guidance: "quantum models are most likely to add value when features ≤ 16, samples ≤ 500, and geometric difference score > 1.0"

### Task 12: docs/config_reference.md

**Auto-generated from Pydantic schema.** Write a script:

```python
# scripts/generate_config_docs.py
from claryon.config_schema import ClaryonConfig
schema = ClaryonConfig.model_json_schema()
# Walk schema, generate markdown table per section
```

Every YAML key with: type, default, description, valid values.

### Task 13: docs/user_guide.md

Step-by-step tutorial:
1. Installation
2. Prepare your data (CSV format, NIfTI layout)
3. Write a config file (with annotated example)
4. Run an experiment
5. Read the results (metrics_summary.csv, methods.tex, SHAP plots)
6. Run inference on new patients
7. Troubleshooting (common errors and fixes)

### Task 14: docs/architecture.md

For developers:
1. Module dependency diagram (text-based, mermaid syntax)
2. Data flow through the pipeline (stage by stage)
3. Registry pattern explanation
4. How preprocessing state flows through folds
5. How to add a new model, metric, explainer, or encoding

### Task 15: docs/contributor_guide.md

1. Development setup
2. Running tests
3. Adding a new model (with template)
4. Adding a new metric
5. PR checklist
6. Code style rules

## ============================================================
## GROUP E: CLEANUP
## ============================================================

### Task 16: Remove Claude session files from repo

```bash
git rm CLAUDE.md COMMANDS.md IMPLEMENTATION_PLAN.md WORKLOG.md
# These are session orchestration files, not part of the software
# Move to a docs/development/ directory if you want to preserve them
```

Actually: move them to `docs/development/` for historical reference, but they should NOT be in the repo root for published releases.

### Task 17: Remove source_archive/

All ports are complete and verified. Source archive is no longer needed.

```bash
git rm -r source_archive/
```

Add `source_archive/` to `.gitignore` in case it gets recreated locally.

### Task 18: Cleanup build artifacts + .gitignore

```bash
git rm -r catboost_info/ claryon.egg-info/ 2>/dev/null
rm -f validation_unit.log validation_full.log fix_log.txt
```

Update `.gitignore`:
```
catboost_info/
*.egg-info/
fix_log.txt
CLAUDE.md
WORKLOG.md
COMMANDS.md
BLOCKED
*.log
!scripts/*.sh
```

### Task 19: Update README.md

- Add presets section (quick/small/medium/large/exhaustive/auto)
- Add inference section (`claryon infer`)
- Add Geometric Difference section
- Update quantum models table with all new models
- Add Huang2021 to references
- Update version number
- Update feature count
- Author: Laszlo Papp, PhD, EANM AI Committee member

### Task 20: Final integration test

Run ALL demo configs end-to-end:
```bash
python -m claryon -v run -c configs/iris_classical.yaml
python -m claryon -v run -c configs/iris_quantum.yaml
python -m claryon -v run -c configs/iris_full_preprocess.yaml
python -m claryon -v run -c configs/iris_quantum_full.yaml
python -m claryon -v run -c configs/nifti_cnn.yaml
```

Verify for each:
- Predictions.csv exists for all models/folds
- preprocessing_state.json exists per fold
- model saved to disk
- metrics_summary.csv generated
- methods.tex is structured prose (not stub)
- results.tex has metrics table
- SHAP/LIME plots generated (where configured)
- run_info.json generated
- No crashes, no uncaught exceptions

Then:
```bash
python -m pytest tests/ -q --timeout=600
git tag v0.11.0-polished
```

## ============================================================
## WORKFLOW
## ============================================================
##
## Execute groups A→B→C→D→E in order.
## Within each group, execute tasks in numbered order.
## After EACH task:
##   1. Run: python -m pytest tests/ -x -q --timeout=300
##   2. If tests pass: git add -A && git commit -m "<group>: <description>"
##   3. If tests fail: fix loop (max 5 attempts)
##
## After each GROUP:
##   Run a quick integration test:
##   python -m claryon -v run -c configs/iris_full_preprocess.yaml
##   Verify no regressions.
##
## ============================================================

## Code rules

- `from __future__ import annotations`
- Type hints, Google docstrings, logging not print
- `@register` decorators where applicable
- Deterministic seeding
- Predictions through io/predictions.py, separator `;`
- **Never crash on resource limits — log error, skip model, continue**

## Fix loop / stop conditions

Same as always. 5 attempts per error, then BLOCKED file.

## Key references

- Huang, H.-Y. et al. "Power of data in quantum machine learning." Nature Communications 12, 2631 (2021). doi:10.1038/s41467-021-22539-9
- Moradi S, ..., Papp L. Sci Rep 12, 1851 (2022). doi:10.1038/s41598-022-05971-9
- Moradi S, ..., Papp L. Eur J Nucl Med Mol Imaging 50, 3826-3837 (2023). doi:10.1007/s00259-023-06362-6
- Papp L, et al. "Quantum Convolutional Neural Networks for Predicting ISUP Grade risk in [68Ga]Ga-PSMA Primary Prostate Cancer Patients." Under revision.
