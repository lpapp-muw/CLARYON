# CLAUDE.md — Autonomous Build Orchestration

## Project

**CLARYON** — CLassical-quantum AI for Reproducible Explainable OpeN-source medicine.
Autonomous build with benchmark-gated progression. No human in the loop unless a stop condition fires.

## Governing documents (read in this order)

1. **WORKLOG.md** — Current state, hard facts, next steps. READ THIS FIRST.
2. **IMPLEMENTATION_PLAN.md** — Build order, file mappings, dependency chains, phase gates.
3. **REQUIREMENTS.md** — Authoritative spec (what to build).

## Source reference

Original source files live in `source_archive/`. Two subdirectories:

- `source_archive/benchmark/` — DEBI-NN benchmark harness (16 files)
- `source_archive/eanm_ai_qc/` — EANM-AI-QC quantum ML framework (25+ files)

When IMPLEMENTATION_PLAN.md says PORT or REWRITE from [B] or [E], read the corresponding source file from `source_archive/` BEFORE writing any code. Do not guess at original logic.

## ============================================================
## MASTER WORKFLOW LOOP
## ============================================================
##
## BEFORE doing anything:
##   1. Read WORKLOG.md → find CURRENT_PHASE and CURRENT_STEP
##   2. Read IMPLEMENTATION_PLAN.md Section 5 → find build order for current phase
##   3. Resume from the next incomplete substep. Never re-implement completed steps.
##
## For each implementation step:
##
##   1.  PLAN
##       Identify files to create/modify (1-3 max per substep).
##       Write plan to stdout before starting.
##
##   2.  CHECK DEPENDENCIES
##       Read IMPLEMENTATION_PLAN.md Section 3 (dependency chains).
##       Verify every import will resolve to an existing module.
##       If a dependency doesn't exist yet, STOP — do not proceed.
##
##   3.  READ SOURCE
##       If disposition is PORT or REWRITE:
##         cat source_archive/benchmark/<file> or source_archive/eanm_ai_qc/<file>
##       Understand the logic before writing new code.
##
##   4.  IMPLEMENT
##       Write production code + unit tests in the SAME substep.
##       Follow all code rules (see below).
##       Test file goes in tests/ mirroring src/ structure.
##
##   5.  UNIT TEST
##       python -m pytest tests/ -x -q --timeout=300
##       If tests pass → proceed to step 6.
##       If tests fail → enter FIX LOOP (see below).
##
##   6.  VALIDATION
##       bash scripts/run_validation.sh
##       This runs the appropriate tier based on CURRENT_PHASE.
##       If validation passes → proceed to step 7.
##       If validation fails → enter FIX LOOP (see below).
##
##   7.  COMMIT
##       git add -A
##       git commit -m "P<phase>.<step>: <imperative description>"
##
##   8.  UPDATE STATE
##       Append substep completion to WORKLOG.md Section 2.
##       Update WORKLOG.md Section 1 (Current State) with new CURRENT_STEP.
##       git add WORKLOG.md && git commit -m "update WORKLOG: P<phase>.<step> done"
##
##   9.  PHASE GATE CHECK
##       If this was the last substep in a phase:
##         a. Read IMPLEMENTATION_PLAN.md Section 4 for gate criteria.
##         b. Verify every gate item.
##         c. If gate passes:
##            - git tag v0.<phase>.0
##            - Update WORKLOG.md CURRENT_PHASE to next phase
##            - Commit WORKLOG.md
##         d. If gate fails:
##            - Enter FIX LOOP targeting the failed gate item.
##
##  10.  NEXT STEP
##       Go to step 1 for next substep.
##
## ============================================================

## Fix loop protocol

When tests, validation, or phase gate checks fail:

1. Read the FULL error output. Do not guess.
2. Identify root cause. Log it:
   `echo "FIX_ATTEMPT <n> [P<phase>.<step>]: <diagnosis>" >> fix_log.txt`
3. Apply minimal targeted fix. Do NOT rewrite unrelated code.
4. Re-run the failing command.
5. If fixed: proceed. Keep fix_log.txt for reference.
6. If still failing after 5 attempts on the SAME error:
   - `git add -A && git commit -m "BLOCKED: P<phase>.<step> — <summary>"`
   - Append blocker to WORKLOG.md under ## BLOCKERS section:
     ```
     ### BLOCKER: P<phase>.<step>
     **Error**: <paste exact error>
     **Attempts**: <list what was tried>
     **Root cause hypothesis**: <best guess>
     **Files involved**: <list>
     ```
   - `git add WORKLOG.md && git commit -m "WORKLOG: blocked at P<phase>.<step>"`
   - `touch BLOCKED`
   - STOP. Do not proceed to next step. Wait for human.

## Code rules (apply to EVERY file)

- `from __future__ import annotations` at top of every .py file
- Type hints on all function signatures
- Google-style docstrings on all public functions and classes
- Registry pattern: `@register("model", "xgboost")` etc. for models, metrics, explainers, encodings
- Prediction contract: all model outputs go through `claryon/io/predictions.py`. Models never write CSVs directly.
- CSV separator: `;` everywhere (REQ §8.4)
- Logging: `import logging; logger = logging.getLogger(__name__)` — never use `print()`
- Determinism: every stochastic operation receives a seed. No unseeded randomness.
- Imports: relative within `claryon/` package (e.g., `from .registry import register`)

## Test rules

- Every module gets a test file in the SAME substep it's created.
- Test file mirrors source structure: `claryon/io/nifti.py` → `tests/test_io/test_nifti.py`
- Integration tests go in `tests/test_integration/`
- Use fixtures from `tests/conftest.py` — do NOT regenerate data in individual tests.
- Timeout: 300 seconds per test (set via `pytest --timeout=300`)
- Quantum model smoke tests: ≤4 qubits, ≤2 epochs, ≤20 samples.
- CNN smoke tests: 3 layers max, ≤5 epochs, synthetic data only.
- PyRadiomics tests: firstorder + glcm only, ≤5 volumes.

## Validation tiers (scripts/run_validation.sh handles this automatically)

| Phase | What runs |
|-------|-----------|
| 0     | Unit tests only |
| 1     | Unit tests + NIfTI loader tests + pyradiomics extraction test |
| 2     | All above + classical model smoke tests + quick benchmark (2 datasets, 2 folds) |
| 3     | All above + quantum encoding tests + quantum model smoke + NIfTI→qCNN integration + NIfTI→radiomics→classical integration |
| 4     | All above + CNN 2D/3D smoke + late fusion integration + extended benchmark (5 datasets, 5 folds) |
| 5     | All above + explainability tests (SHAP on 1 classical + 1 quantum model) |
| 6     | All above + metrics tests + statistical comparator tests + report generation tests |
| 7     | Full integration suite + full benchmark (all 28 datasets) |

## Git conventions

- Branch: `main` (linear history, no branches)
- Commit format: `P<phase>.<step>: <imperative description>`
- WORKLOG updates: `update WORKLOG: P<phase>.<step> done`
- Blockers: `BLOCKED: P<phase>.<step> — <summary>`
- Phase tags: `v0.<phase>.0`

## Stop conditions (hard stop, wait for human)

- `BLOCKED` sentinel file created (5 fix attempts exhausted)
- Phase gate failure after fix attempts exhausted
- Architectural ambiguity: two valid approaches, REQUIREMENTS.md doesn't resolve it
- External dependency failure that retries can't fix (pip, system lib)
- Any proposed change to REQUIREMENTS.md or IMPLEMENTATION_PLAN.md itself
- Circular dependency detected
- Test fixture data missing or corrupted

## Resume protocol

When restarting after a stop:

1. Check for `BLOCKED` file. If exists → read WORKLOG.md ## BLOCKERS section.
2. Check `fix_log.txt`. If exists → read for context.
3. If no `BLOCKED` file → read WORKLOG.md Section 1 for CURRENT_STEP.
4. Resume from that exact point. Do not re-run completed steps.
5. Do not re-generate test fixtures (they're committed).

## Fixture data

Pre-generated synthetic data lives in `tests/fixtures/data/`:

- `tabular_binary_train.csv`, `tabular_binary_test.csv` — 80+20 samples, 10 features
- `tabular_multiclass_train.csv`, `tabular_multiclass_test.csv` — 120+30 samples, 10 features, 3 classes
- `tabular_regression_train.csv`, `tabular_regression_test.csv` — 80+20 samples, 10 features
- `nifti_masked/Train/`, `nifti_masked/Test/` — NIfTI volumes with binary masks
- `nifti_nomask/Train/`, `nifti_nomask/Test/` — NIfTI volumes without masks
- `nifti_multilabel/` — NIfTI with integer-labelled mask (3 ROIs)
- `pyradiomics_minimal.yaml` — Minimal pyradiomics config (firstorder + glcm)
- `fdb_ldb/` — Small FDB + LDB files for legacy format testing
- `tiff_synthetic/` — Synthetic TIFF files with metadata sidecar

These are committed and should NOT be regenerated. If corrupt, run:
`python tests/fixtures/generate_fixtures.py`

## DEBI-NN binary

The DEBI-NN C++ binary is NOT included. The `debinn_.py` wrapper will be tested with a mock binary (a simple script that mimics the expected output format). The mock lives at `tests/fixtures/mock_debinn.sh`. Claude Code must create this mock during Phase 2.

## Key hard facts from WORKLOG.md (quick reference)

- HF-003: All CSVs use `;` separator. Float format: `%.8f`. Key format: `S0000..S{n-1}`.
- HF-004: Quantum models produce probabilities clustered near 0.5. Must use Youden's J threshold.
- HF-005: DEBI-NN invoked via subprocess, reads `executionSettings.csv`, needs `QT_QPA_PLATFORM=offscreen`.
- HF-006: 306 features → pad 512 → 9 qubits. Kernel SVM builds N² matrix. Warn on >20 qubits.
- HF-007: Datasets >10K samples use fixed 60/20/20 split instead of k-fold.
