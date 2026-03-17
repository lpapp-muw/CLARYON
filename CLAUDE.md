# CLAUDE.md — Bugfix Session

## Project

**CLARYON** — CLassical-quantum AI for Reproducible Explainable OpeN-source medicine.
This session fixes all known bugs identified from results analysis and testing.

## READ FIRST

1. **WORKLOG.md** — Current state
2. **claryon/pipeline.py** — Pipeline orchestrator
3. **claryon/evaluation/metrics.py** — Metrics module

## ============================================================
## BUGS TO FIX (execute in order)
## ============================================================

### Bug 1: Multiclass AUC returns NaN

**Symptom**: In iris_classical (3-class), metrics_summary.csv shows `;;;` for AUC columns. report.md shows `nan`.

**Root cause**: The AUC metric likely calls `sklearn.metrics.roc_auc_score` without `multi_class="ovr"` or `average="weighted"`. For multiclass problems, sklearn requires these parameters explicitly; otherwise it raises an error or returns NaN.

**Fix location**: `claryon/evaluation/metrics.py` — find the registered `auc` metric function.

**Fix**:
```python
def auc_score(y_true, y_prob, **kwargs):
    n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2
    if n_classes > 2:
        return roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    else:
        # Binary: use probability of positive class
        probs = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        return roc_auc_score(y_true, probs)
```

Also wrap in try/except — if AUC cannot be computed (e.g., only one class present in a fold), return NaN gracefully with a warning, not an empty string.

**Test**: Run `python -m claryon -v run -c configs/iris_classical.yaml` and verify AUC is a valid float in metrics_summary.csv.

### Bug 2: Structured methods.tex not wired after preprocessing rewrite

**Symptom**: iris_preprocess and iris_classical produce the old simple template methods.tex (just model names in bold, no prose paragraphs), despite the structured report generator existing at `claryon/reporting/structured_report.py`.

**Root cause**: The preprocessing session (Phase 10) rewrote `stage_report()` in pipeline.py. The rewrite likely reverted to the simple template instead of calling `generate_structured_methods()`.

**Diagnosis**: Read `claryon/pipeline.py` function `stage_report()`. Check if it imports and calls `structured_report.generate_structured_methods()`. If not, wire it.

**Fix**: Ensure `stage_report()` calls:
```python
from .reporting.structured_report import generate_structured_methods
generate_structured_methods(
    config, results_dir / "methods.tex",
    n_samples=n_samples, n_features=n_features,
)
```
with fallback to simple template on import/call error.

**Test**: Run iris_preprocess config. Check that methods.tex contains prose paragraphs (subsections like "Experimental Setup", "Data", "Models" with full sentences), not just bold model names.

### Bug 3: Float precision in metrics_summary.csv

**Symptom**: Values like `0.9949999999999999` instead of `0.995`, and `0.010000000000000009` instead of `0.01`.

**Root cause**: Python float representation written directly without rounding.

**Fix location**: Where metrics_summary.csv is written — likely in `claryon/evaluation/results_store.py` or in `stage_evaluate()` in pipeline.py.

**Fix**: Round all metric values to 6 decimal places before writing:
```python
value = round(value, 6)
```

Or use formatted string: `f"{value:.6f}"`.

Apply to both mean and std columns. Do NOT round inside the metric computation itself — only at the output/display stage.

**Test**: Run any experiment. Verify metrics_summary.csv values are clean (e.g., `0.995` not `0.9949999999999999`).

### Bug 4: metrics_summary.csv has empty fields instead of NaN

**Symptom**: iris_classical AUC columns show `;;;` (empty) instead of `nan` or `N/A`.

**Root cause**: When AUC computation fails, the value is stored as empty string or None, which serializes as empty in CSV.

**Fix**: If a metric cannot be computed, store `float('nan')`. When writing CSV, format NaN as `NaN` (not empty). This keeps the CSV parseable by pandas (`pd.read_csv` handles `NaN` correctly).

**Fix location**: Same as Bug 1 and Bug 3 — metrics computation + CSV writing.

### Bug 5: iris_full has no preprocessing_state.json

**Symptom**: `find Results/iris_full -name "preprocessing_state.json"` returns 0 files.

**Root cause**: iris_full was run BEFORE the preprocessing session. The old pipeline didn't save preprocessing state. This is not a bug in the current code — it's expected for old results.

**Fix**: Not a code fix. But verify that running iris_full NOW (with the current pipeline) DOES produce preprocessing_state.json files. If it doesn't, then the preprocessing wiring only activates when `preprocessing` is explicitly in the config.

**Diagnosis**: Check if the default `PreprocessingConfig` (zscore=True, feature_selection=True) is applied when the user doesn't specify `preprocessing:` in their YAML. If the config field has a default factory, it should apply automatically.

**Test**: Delete Results/iris_full and re-run `python -m claryon -v run -c configs/iris_full.yaml`. Verify preprocessing_state.json appears in every fold directory.

### Bug 6: Ensemble row appears in iris_preprocess but ensemble was not configured

**Symptom**: iris_preprocess results.tex shows an "Ensemble" row with metrics, but the config doesn't have an `ensemble:` section.

**Root cause**: The pipeline may auto-generate an ensemble row whenever there are 2+ models. This might be intentional (auto-ensemble) or a bug (ensemble should only appear when configured).

**Diagnosis**: Read pipeline.py stage_evaluate or stage_report to understand when the Ensemble row is added. If it's always-on, that's a design decision. If it should be config-controlled, add a check.

**Decision**: If ensemble is always-on, document it. If it should be optional, gate it behind:
```yaml
ensemble:
  enabled: true
  strategy: softmax_average
```

**For now**: Leave as-is if it's intentional auto-ensemble. Just ensure the Ensemble row only appears in results.tex when there are 2+ models, and that the ensemble metrics are correct (average of individual model predictions, not average of metrics).

### Bug 7: report.md does not include std values

**Symptom**: report.md shows `| xgboost | 0.9950 | 0.9950 | 0.9900 | 1.0000 |` without standard deviations. The metrics_summary.csv HAS std columns but the markdown report drops them.

**Fix location**: `claryon/reporting/markdown_report.py`

**Fix**: Include std in the table, formatted as `mean ± std`:
```
| xgboost | 0.995 ± 0.010 | 0.995 ± 0.010 | 0.990 ± 0.020 | 1.000 ± 0.000 |
```

Or as a separate column. The `±` format is more compact and standard in medical literature.

### Bug 8: results.tex does not include std values

**Symptom**: Same as Bug 7 but for LaTeX. The results table shows bare values without ± std.

**Fix location**: `claryon/reporting/latex_report.py` — the RESULTS_TEMPLATE.

**Fix**: Format as `0.995 $\pm$ 0.010` in LaTeX:
```latex
xgboost & 0.995 $\pm$ 0.010 & 0.995 $\pm$ 0.010 & ...
```

## ============================================================
## WORKFLOW
## ============================================================
##
## For each bug:
##   1. Read the relevant source file
##   2. Apply minimal fix
##   3. Run: python -m pytest tests/ -x -q --timeout=300
##   4. Run the relevant experiment config to verify fix
##   5. Commit: git add -A && git commit -m "fix: <description>"
##
## After all bugs:
##   1. Run full test suite
##   2. Run ALL demo experiments:
##      python -m claryon -v run -c configs/iris_classical.yaml
##      python -m claryon -v run -c configs/iris_full.yaml
##      python -m claryon -v run -c configs/iris_full_preprocess.yaml
##      python -m claryon -v run -c configs/iris_quantum.yaml
##   3. Verify:
##      - No NaN in AUC for binary experiments
##      - Multiclass AUC is a valid float (not empty)
##      - metrics_summary.csv values are rounded (6 decimal places)
##      - methods.tex has structured prose (not just bold names)
##      - results.tex and report.md show mean ± std
##      - preprocessing_state.json in every fold
##   4. git tag v0.11.1-bugfix
##   5. Update WORKLOG.md
##
## ============================================================

## Code rules

Same as always:
- `from __future__ import annotations`
- Type hints, Google docstrings, logging not print
- Minimal fixes — do not refactor unrelated code
- Wrap metric computations in try/except, never crash on a single metric failure

## Fix loop protocol

5 attempts per bug. If stuck, commit partial fix and move to next bug.
