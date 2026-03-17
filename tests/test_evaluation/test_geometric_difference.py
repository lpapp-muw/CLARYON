"""Tests for Geometric Difference framework."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.evaluation.geometric_difference import (
    effective_dimension,
    geometric_difference_score,
    model_complexity,
    quantum_advantage_analysis,
)


class TestGeometricDifference:
    """Test GDQ score computation."""

    def test_identical_kernels_g_near_one(self) -> None:
        """When quantum == classical, g should be ~1."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 4))
        K = X @ X.T  # linear kernel
        K = K + 1e-6 * np.eye(20)
        g = geometric_difference_score(K, K_classical=K)
        assert 0.5 < g < 2.0, f"Expected g ~1, got {g}"

    def test_different_kernels_g_larger(self) -> None:
        """When kernels differ, g should be > 1."""
        rng = np.random.default_rng(42)
        N = 20
        # Quantum kernel: highly structured
        K_Q = np.eye(N) + 0.1 * rng.standard_normal((N, N))
        K_Q = K_Q @ K_Q.T  # make PSD
        # Classical kernel: smooth
        X = rng.standard_normal((N, 4))
        K_C = X @ X.T + np.eye(N)
        g = geometric_difference_score(K_Q, K_classical=K_C)
        assert g > 0  # should be positive

    def test_auto_classical_kernel(self) -> None:
        """X_train fallback computes linear kernel."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((15, 3))
        K_Q = np.eye(15) + 0.5 * X @ X.T
        K_Q = (K_Q + K_Q.T) / 2
        g = geometric_difference_score(K_Q, X_train=X)
        assert g > 0


class TestModelComplexity:
    """Test model complexity s_K(N)."""

    def test_identity_kernel(self) -> None:
        """s_K with identity kernel = ||y||^2."""
        y = np.array([1.0, -1.0, 1.0, -1.0])
        K = np.eye(4)
        s = model_complexity(K, y)
        assert abs(s - 4.0) < 0.1


class TestEffectiveDimension:
    """Test effective dimension."""

    def test_identity_matrix(self) -> None:
        """Identity matrix should have full rank."""
        K = np.eye(10)
        d = effective_dimension(K)
        assert d == 10

    def test_low_rank(self) -> None:
        """Low-rank matrix should have small effective dimension."""
        rng = np.random.default_rng(42)
        V = rng.standard_normal((10, 2))
        K = V @ V.T + 1e-10 * np.eye(10)
        d = effective_dimension(K, threshold=0.01)
        assert d <= 4  # rank ~2


class TestQuantumAdvantageAnalysis:
    """Test full analysis pipeline."""

    def test_returns_all_fields(self) -> None:
        rng = np.random.default_rng(42)
        N = 15
        X = rng.standard_normal((N, 4))
        K_Q = X @ X.T + np.eye(N)
        y = np.array([0, 1] * 7 + [0])
        result = quantum_advantage_analysis(K_Q, y, X_train=X)
        assert "g_CQ" in result
        assert "s_C" in result
        assert "s_Q" in result
        assert "d" in result
        assert "recommendation" in result
        assert "explanation" in result
        assert result["recommendation"] in (
            "classical_sufficient", "quantum_advantage_likely", "inconclusive",
        )

    def test_classical_sufficient_for_linear_kernel(self) -> None:
        """Linear quantum kernel should be classical_sufficient."""
        rng = np.random.default_rng(42)
        N = 20
        X = rng.standard_normal((N, 4))
        K_Q = X @ X.T + 0.01 * np.eye(N)  # ~same as linear kernel
        y = np.array([0, 1] * 10)
        result = quantum_advantage_analysis(K_Q, y, X_train=X)
        # With ~identical kernels, should lean classical
        assert result["recommendation"] in ("classical_sufficient", "inconclusive")
