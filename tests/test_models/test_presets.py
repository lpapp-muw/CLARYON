"""Tests for model preset resolution."""
from __future__ import annotations

import pytest

from claryon.models.preset_resolver import resolve_model_params, resolve_preset


class TestResolvePreset:
    """Test preset loading and category defaults."""

    def test_tabular_medium(self) -> None:
        params = resolve_preset("xgboost", "tabular", "medium")
        assert "n_estimators" in params
        assert params["n_estimators"] == 500

    def test_mlp_no_tree_params(self) -> None:
        params = resolve_preset("mlp", "tabular", "medium")
        assert "n_estimators" not in params
        assert "max_iter" in params

    def test_quantum_quick(self) -> None:
        params = resolve_preset("qcnn_muw", "tabular_quantum", "quick")
        assert params["epochs"] == 5

    def test_model_override_merges(self) -> None:
        params = resolve_preset("qcnn_muw", "tabular_quantum", "medium")
        # Model override sets init_scale
        assert params["init_scale"] == 0.1
        # Category default sets batch_size
        assert params["batch_size"] == 16

    def test_null_values_removed(self) -> None:
        params = resolve_preset("kernel_svm", "tabular_quantum", "quick")
        assert "shots" not in params

    def test_imaging_preset(self) -> None:
        params = resolve_preset("cnn_3d", "imaging", "large")
        assert params["epochs"] == 100
        assert params["batch_size"] == 4


class TestResolveModelParams:
    """Test full resolution priority."""

    def test_explicit_params_override_preset(self) -> None:
        params = resolve_model_params(
            model_name="xgboost",
            model_type="tabular",
            explicit_params={"n_estimators": 42},
            model_preset="large",
            global_complexity="medium",
        )
        assert params["n_estimators"] == 42
        # Other params from preset still present
        assert params["max_depth"] == 10  # from large preset

    def test_model_preset_overrides_global(self) -> None:
        params = resolve_model_params(
            model_name="xgboost",
            model_type="tabular",
            explicit_params={},
            model_preset="quick",
            global_complexity="large",
        )
        assert params["n_estimators"] == 50  # quick, not large

    def test_global_complexity_used_when_no_model_preset(self) -> None:
        params = resolve_model_params(
            model_name="xgboost",
            model_type="tabular",
            explicit_params={},
            model_preset=None,
            global_complexity="large",
        )
        assert params["n_estimators"] == 1000

    def test_defaults_to_medium(self) -> None:
        params = resolve_model_params(
            model_name="xgboost",
            model_type="tabular",
            explicit_params={},
            model_preset=None,
            global_complexity="auto",
        )
        assert params["n_estimators"] == 500  # medium default

    def test_quantum_low_preset_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        with caplog.at_level(logging.WARNING):
            resolve_model_params(
                model_name="qcnn_muw",
                model_type="tabular_quantum",
                explicit_params={},
                model_preset="quick",
                global_complexity="medium",
            )
        assert "quick" in caplog.text
        assert "publishable" in caplog.text.lower() or "WARNING" in caplog.text
