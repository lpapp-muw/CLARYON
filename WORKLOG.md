# CLARYON — Work Log

**Purpose**: Cross-session continuity document. Read this FIRST.

---

## 1. Current State

| Field | Value |
|---|---|
| **Phase** | 11.1 — bugfixes |
| **Last completed item** | Polish session (presets, auto mode, GDQ, CLI, docs, cleanup) |
| **Next item to build** | Fix 8 known bugs from results analysis |
| **Blockers** | None |
| **CURRENT_PHASE** | 11.1 |
| **CURRENT_STEP** | bugfix |

---

## 2. Known Bugs

| # | Bug | Severity | Location |
|---|---|---|---|
| 1 | Multiclass AUC returns NaN | Medium | evaluation/metrics.py |
| 2 | Structured methods.tex not wired after preprocessing rewrite | Medium | pipeline.py stage_report |
| 3 | Float precision in metrics_summary.csv (0.9949999999 instead of 0.995) | Low | metrics CSV writer |
| 4 | Empty fields instead of NaN in metrics_summary.csv | Low | metrics CSV writer |
| 5 | iris_full missing preprocessing_state.json (verify current pipeline produces it) | Low | pipeline.py |
| 6 | Ensemble row auto-appears without config (verify if intentional) | Low | pipeline.py or stage_evaluate |
| 7 | report.md missing std values | Low | reporting/markdown_report.py |
| 8 | results.tex missing std values | Low | reporting/latex_report.py |

---

## 3. Hard Facts

All previous HF-001 through HF-030 apply. No new hard facts for this session.
