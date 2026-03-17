# CLARYON — Work Log

**Purpose**: Cross-session continuity document. Read this FIRST.

---

## 1. Current State

| Field | Value |
|---|---|
| **Phase** | 11.1 — bugfixes (COMPLETE) |
| **Last completed item** | Fixed all 8 known bugs from results analysis |
| **Next item to build** | TBD |
| **Blockers** | None |
| **CURRENT_PHASE** | 11.1 |
| **CURRENT_STEP** | done |

---

## 2. Known Bugs — ALL FIXED (v0.11.1-bugfix)

| # | Bug | Status | Fix |
|---|---|---|---|
| 1 | Multiclass AUC returns NaN | FIXED | `multi_class="ovr", average="weighted"` for >2 classes |
| 2 | Structured methods.tex not wired | FIXED | Removed duplicate simple-template overwrite in stage_report |
| 3 | Float precision in metrics_summary.csv | FIXED | Round numeric columns to 6 decimal places |
| 4 | Empty fields instead of NaN | FIXED | `na_rep="NaN"` in CSV writer |
| 5 | iris_full missing preprocessing_state.json | VERIFIED | Default PreprocessingConfig produces it (70 files for iris_full) |
| 6 | Ensemble row auto-appears without config | BY DESIGN | Auto-ensemble for 2+ models in LaTeX results only |
| 7 | report.md missing std values | FIXED | `mean ± std` format via format_metric helper |
| 8 | results.tex missing std values | FIXED | `mean $\pm$ std` format via format_metric_latex helper |

---

## 3. Hard Facts

All previous HF-001 through HF-030 apply. No new hard facts for this session.
