"""Tests for auto complexity mode."""
from __future__ import annotations

from claryon.config_schema import ClaryonConfig, ModelEntry
from claryon.models.auto_complexity import auto_select_presets, estimate_runtime


class TestEstimateRuntime:
    """Test runtime estimation."""

    def test_classical_instant(self) -> None:
        est = estimate_runtime("xgboost", "tabular", 1000, 0, {})
        assert est == 5.0

    def test_quantum_kernel_scales_quadratic(self) -> None:
        est_100 = estimate_runtime("kernel_svm", "tabular_quantum", 100, 4, {})
        est_200 = estimate_runtime("kernel_svm", "tabular_quantum", 200, 4, {})
        assert est_200 > est_100 * 3  # ~4x for N²

    def test_more_qubits_slower(self) -> None:
        est_4 = estimate_runtime("qcnn_muw", "tabular_quantum", 50, 4, {"epochs": 10, "batch_size": 16})
        est_8 = estimate_runtime("qcnn_muw", "tabular_quantum", 50, 8, {"epochs": 10, "batch_size": 16})
        assert est_8 > est_4


class TestAutoSelectPresets:
    """Test auto preset selection."""

    def test_classical_gets_at_least_medium(self) -> None:
        config = ClaryonConfig(
            models=[ModelEntry(name="xgboost", type="tabular")],
            experiment={"name": "test", "complexity": "auto", "max_runtime_minutes": 1},
        )
        selected = auto_select_presets(config, n_samples=100, n_features=10, n_features_after_mrmr=4)
        assert selected["xgboost"] in ("medium", "large", "exhaustive")

    def test_tight_budget_selects_lower(self) -> None:
        config = ClaryonConfig(
            models=[ModelEntry(name="qcnn_muw", type="tabular_quantum")],
            experiment={"name": "test", "complexity": "auto", "max_runtime_minutes": 1},
        )
        # 10 qubits = large state vector, should force lower preset
        selected = auto_select_presets(config, n_samples=100, n_features=1024, n_features_after_mrmr=1024)
        assert selected["qcnn_muw"] in ("quick", "small", "medium")

    def test_generous_budget_selects_higher(self) -> None:
        config = ClaryonConfig(
            models=[ModelEntry(name="qcnn_muw", type="tabular_quantum")],
            experiment={"name": "test", "complexity": "auto", "max_runtime_minutes": 600},
        )
        selected = auto_select_presets(config, n_samples=20, n_features=4, n_features_after_mrmr=4)
        # Small dataset + big budget → should pick high preset
        assert selected["qcnn_muw"] in ("large", "exhaustive")
