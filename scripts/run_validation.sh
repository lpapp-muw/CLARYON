#!/usr/bin/env bash
# =============================================================================
# CLARYON — Tiered Validation Runner
# Called by Claude Code after every implementation step.
# Selects appropriate test tier based on CURRENT_PHASE in WORKLOG.md.
# Exit 0 = pass, Exit 1 = fail.
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate 2>/dev/null || true
fi

echo "=== VALIDATION START: $(date) ==="
echo "=== Project: $PROJECT_DIR ==="

# ── Determine current phase ──
PHASE=$(grep -oP 'CURRENT_PHASE:\s*\K\d+' WORKLOG.md 2>/dev/null || echo "0")
echo "=== Current phase: $PHASE ==="

FAIL=0

# ──────────────────────────────────────────────────
# TIER 1: Unit tests (ALWAYS)
# ──────────────────────────────────────────────────
echo ""
echo "--- TIER 1: Unit tests ---"
python -m pytest tests/ -x -q --timeout=300 \
    --ignore=tests/test_integration 2>&1 | tee validation_unit.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "FAIL: Unit tests"
    FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
    echo "=== VALIDATION FAILED at TIER 1: $(date) ==="
    exit 1
fi

# ──────────────────────────────────────────────────
# TIER 2: I/O + Preprocessing (Phase 1+)
# ──────────────────────────────────────────────────
if [ "$PHASE" -ge 1 ]; then
    echo ""
    echo "--- TIER 2a: NIfTI loader + mask pipeline ---"
    if [ -f "tests/test_io/test_nifti.py" ]; then
        python -m pytest tests/test_io/test_nifti.py -x -q --timeout=300 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 2b: Tabular loader ---"
    if [ -f "tests/test_io/test_tabular.py" ]; then
        python -m pytest tests/test_io/test_tabular.py -x -q --timeout=300 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 2c: FDB/LDB loader ---"
    if [ -f "tests/test_io/test_fdb_ldb.py" ]; then
        python -m pytest tests/test_io/test_fdb_ldb.py -x -q --timeout=300 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 2d: Splits + preprocessing ---"
    if [ -d "tests/test_preprocessing" ]; then
        python -m pytest tests/test_preprocessing/ -x -q --timeout=300 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 2e: PyRadiomics extraction ---"
    if [ -f "tests/test_preprocessing/test_radiomics.py" ]; then
        python -m pytest tests/test_preprocessing/test_radiomics.py -x -q --timeout=600 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 2f: Encoding ---"
    if [ -d "tests/test_encoding" ]; then
        python -m pytest tests/test_encoding/ -x -q --timeout=300 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    if [ "$FAIL" -ne 0 ]; then
        echo "=== VALIDATION FAILED at TIER 2: $(date) ==="
        exit 1
    fi
fi

# ──────────────────────────────────────────────────
# TIER 3: Classical models + benchmark (Phase 2+)
# ──────────────────────────────────────────────────
if [ "$PHASE" -ge 2 ]; then
    echo ""
    echo "--- TIER 3a: Classical model smoke tests ---"
    if [ -d "tests/test_models/test_classical" ]; then
        python -m pytest tests/test_models/test_classical/ -x -q --timeout=600 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 3b: Ensemble test ---"
    if [ -f "tests/test_models/test_ensemble.py" ]; then
        python -m pytest tests/test_models/test_ensemble.py -x -q --timeout=300 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 3c: Pipeline integration (classical) ---"
    if [ -f "tests/test_integration/test_classical_pipeline.py" ]; then
        python -m pytest tests/test_integration/test_classical_pipeline.py -x -q --timeout=600 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    # Quick tabular benchmark — only if benchmark module exists
    if [ -f "claryon/benchmark/downloader.py" ] && python -c "import claryon.benchmark" 2>/dev/null; then
        echo ""
        echo "--- TIER 3d: Quick tabular benchmark (2 datasets, 2 folds) ---"
        echo "    This may take 5-15 minutes..."
        python -m claryon benchmark --datasets iris,wine --folds 2 --quick 2>&1 | tee validation_bench_t3.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    if [ "$FAIL" -ne 0 ]; then
        echo "=== VALIDATION FAILED at TIER 3: $(date) ==="
        exit 1
    fi
fi

# ──────────────────────────────────────────────────
# TIER 4: Quantum models + integration (Phase 3+)
# ──────────────────────────────────────────────────
if [ "$PHASE" -ge 3 ]; then
    echo ""
    echo "--- TIER 4a: Quantum encoding tests ---"
    if [ -d "tests/test_encoding" ]; then
        python -m pytest tests/test_encoding/ -x -q --timeout=300 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 4b: Quantum model smoke tests (kernel_svm, qcnn_muw, qcnn_alt) ---"
    if [ -d "tests/test_models/test_quantum" ]; then
        python -m pytest tests/test_models/test_quantum/ -x -q --timeout=900 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 4c: Integration: NIfTI → amplitude encode → qCNN → predictions ---"
    if [ -f "tests/test_integration/test_nifti_qcnn_pipeline.py" ]; then
        python -m pytest tests/test_integration/test_nifti_qcnn_pipeline.py -x -q --timeout=900 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 4d: Integration: NIfTI → pyradiomics → classical model ---"
    if [ -f "tests/test_integration/test_nifti_radiomics_pipeline.py" ]; then
        python -m pytest tests/test_integration/test_nifti_radiomics_pipeline.py -x -q --timeout=600 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    if [ "$FAIL" -ne 0 ]; then
        echo "=== VALIDATION FAILED at TIER 4: $(date) ==="
        exit 1
    fi
fi

# ──────────────────────────────────────────────────
# TIER 5: CNN / 3D CNN + late fusion (Phase 4+)
# ──────────────────────────────────────────────────
if [ "$PHASE" -ge 4 ]; then
    echo ""
    echo "--- TIER 5a: 2D CNN smoke test ---"
    if [ -f "tests/test_models/test_classical/test_cnn_2d.py" ]; then
        python -m pytest tests/test_models/test_classical/test_cnn_2d.py -x -q --timeout=600 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 5b: 3D CNN smoke test on synthetic NIfTI ---"
    if [ -f "tests/test_models/test_classical/test_cnn_3d.py" ]; then
        python -m pytest tests/test_models/test_classical/test_cnn_3d.py -x -q --timeout=600 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 5c: Late fusion integration ---"
    if [ -f "tests/test_integration/test_late_fusion.py" ]; then
        python -m pytest tests/test_integration/test_late_fusion.py -x -q --timeout=600 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    # Extended benchmark
    if [ -f "claryon/benchmark/downloader.py" ] && python -c "import claryon.benchmark" 2>/dev/null; then
        echo ""
        echo "--- TIER 5d: Extended benchmark (5 datasets, 5 folds) ---"
        echo "    This may take 1-4 hours..."
        python -m claryon benchmark \
            --datasets iris,wine,breast_cancer,vehicle,segment \
            --folds 5 2>&1 | tee validation_bench_t5.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    if [ "$FAIL" -ne 0 ]; then
        echo "=== VALIDATION FAILED at TIER 5: $(date) ==="
        exit 1
    fi
fi

# ──────────────────────────────────────────────────
# TIER 6: Explainability (Phase 5+)
# ──────────────────────────────────────────────────
if [ "$PHASE" -ge 5 ]; then
    echo ""
    echo "--- TIER 6a: SHAP tests ---"
    if [ -f "tests/test_explainability/test_shap.py" ]; then
        python -m pytest tests/test_explainability/test_shap.py -x -q --timeout=600 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 6b: LIME tests ---"
    if [ -f "tests/test_explainability/test_lime.py" ]; then
        python -m pytest tests/test_explainability/test_lime.py -x -q --timeout=600 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    if [ "$FAIL" -ne 0 ]; then
        echo "=== VALIDATION FAILED at TIER 6: $(date) ==="
        exit 1
    fi
fi

# ──────────────────────────────────────────────────
# TIER 7: Evaluation + Reporting (Phase 6+)
# ──────────────────────────────────────────────────
if [ "$PHASE" -ge 6 ]; then
    echo ""
    echo "--- TIER 7a: Metrics tests ---"
    if [ -d "tests/test_evaluation" ]; then
        python -m pytest tests/test_evaluation/ -x -q --timeout=300 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    echo ""
    echo "--- TIER 7b: Report generation tests ---"
    if [ -f "tests/test_integration/test_reporting.py" ]; then
        python -m pytest tests/test_integration/test_reporting.py -x -q --timeout=300 2>&1 | tee -a validation_unit.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    if [ "$FAIL" -ne 0 ]; then
        echo "=== VALIDATION FAILED at TIER 7: $(date) ==="
        exit 1
    fi
fi

# ──────────────────────────────────────────────────
# TIER 8: Full benchmark (Phase 7)
# ──────────────────────────────────────────────────
if [ "$PHASE" -ge 7 ]; then
    echo ""
    echo "--- TIER 8: Full integration test suite ---"
    python -m pytest tests/ -q --timeout=600 2>&1 | tee validation_full.log
    [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1

    if [ -f "claryon/benchmark/downloader.py" ] && python -c "import claryon.benchmark" 2>/dev/null; then
        echo ""
        echo "--- TIER 8b: Full benchmark (all datasets) ---"
        echo "    This may take many hours..."
        python -m claryon benchmark 2>&1 | tee validation_bench_full.log
        [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
    fi

    if [ "$FAIL" -ne 0 ]; then
        echo "=== VALIDATION FAILED at TIER 8: $(date) ==="
        exit 1
    fi
fi

echo ""
echo "=== VALIDATION PASSED (Phase $PHASE): $(date) ==="
exit 0
