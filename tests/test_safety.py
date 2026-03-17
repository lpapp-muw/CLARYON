"""Tests for resource estimation and OOM protection."""
from __future__ import annotations

from unittest.mock import patch

from claryon.safety import (
    estimate_memory_gb,
    get_available_memory_gb,
    preflight_resource_check,
)


class TestPreflightResourceCheck:
    """Test preflight safety checks."""

    def test_warns_for_many_qubits(self) -> None:
        warnings = preflight_resource_check(
            "kernel_svm", "tabular_quantum", 100, 21, {},
        )
        assert any("RUNTIME WARNING" in w for w in warnings)
        assert any("21 qubits" in w for w in warnings)

    def test_warns_for_large_kernel(self) -> None:
        warnings = preflight_resource_check(
            "kernel_svm", "tabular_quantum", 20000, 4, {},
        )
        assert any("kernel matrix" in w.lower() for w in warnings)

    def test_warns_for_swap_test(self) -> None:
        warnings = preflight_resource_check(
            "qdc_swap", "tabular_quantum", 100, 16, {},
        )
        assert any("qdc_swap" in w for w in warnings)

    def test_no_warnings_for_small_model(self) -> None:
        warnings = preflight_resource_check(
            "kernel_svm", "tabular_quantum", 50, 3, {},
        )
        assert len(warnings) == 0

    def test_long_runtime_warning(self) -> None:
        warnings = preflight_resource_check(
            "qcnn_muw", "tabular_quantum", 1000, 15,
            {"epochs": 500, "batch_size": 8},
        )
        assert any("hours" in w.lower() for w in warnings)


class TestEstimateMemory:
    """Test memory estimation."""

    def test_state_vector_grows_exponentially(self) -> None:
        mem_4 = estimate_memory_gb("qcnn_muw", 4, 100)
        mem_8 = estimate_memory_gb("qcnn_muw", 8, 100)
        assert mem_8 > mem_4 * 10

    def test_kernel_model_adds_matrix(self) -> None:
        mem = estimate_memory_gb("kernel_svm", 4, 1000)
        assert mem > 0.007  # at least 1000² * 8 bytes ≈ 0.008 GB


class TestGetAvailableMemory:
    """Test memory detection."""

    def test_returns_positive(self) -> None:
        mem = get_available_memory_gb()
        assert mem > 0

    def test_fallback_on_error(self) -> None:
        with patch("builtins.open", side_effect=OSError("no /proc")):
            mem = get_available_memory_gb()
            assert mem > 0  # falls back to psutil or 8.0
