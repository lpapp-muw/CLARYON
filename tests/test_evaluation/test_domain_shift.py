"""Tests for claryon.evaluation.domain_shift — KS, center classifier, MMD, UMAP."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.evaluation.domain_shift import (
    HAS_UMAP,
    center_classifier_bacc,
    ks_pairwise,
    mmd_pairwise,
    umap_by_center,
)


@pytest.fixture
def shifted_data():
    """3 centers, 10 features, 30 samples each.

    A ~ N(0,1), B ~ N(2,1) (shifted), C ~ N(0,1) (same as A).
    """
    rng = np.random.default_rng(42)
    n = 30
    d = 10
    X_a = rng.standard_normal((n, d))
    X_b = rng.standard_normal((n, d)) + 2.0
    X_c = rng.standard_normal((n, d))
    X = np.vstack([X_a, X_b, X_c])
    center_ids = np.array(["A"] * n + ["B"] * n + ["C"] * n)
    return X, center_ids


@pytest.fixture
def identical_data():
    """3 centers, all drawn from the same N(0,1)."""
    rng = np.random.default_rng(99)
    n = 30
    d = 10
    X = rng.standard_normal((3 * n, d))
    center_ids = np.array(["A"] * n + ["B"] * n + ["C"] * n)
    return X, center_ids


# ── KS tests ────────────────────────────────────────────────


def test_ks_detects_shifted_center(shifted_data):
    """A vs B should show high shift %, A vs C should show low."""
    X, center_ids = shifted_data
    results = ks_pairwise(X, center_ids, bonferroni=True)

    ab = results[("A", "B")]
    ac = results[("A", "C")]
    assert ab["pct_shifted"] > ac["pct_shifted"]
    assert ab["pct_shifted"] > 50.0  # B is clearly shifted


def test_ks_bonferroni(shifted_data):
    """With Bonferroni, fewer (or equal) features flagged than without."""
    X, center_ids = shifted_data
    with_bonf = ks_pairwise(X, center_ids, bonferroni=True)
    without_bonf = ks_pairwise(X, center_ids, bonferroni=False)

    for pair in with_bonf:
        assert with_bonf[pair]["n_shifted"] <= without_bonf[pair]["n_shifted"]


# ── Center classifier tests ─────────────────────────────────


def test_center_classifier_separable(shifted_data):
    """BACC well above chance (33%) when B is shifted."""
    X, center_ids = shifted_data
    mean_bacc, std_bacc = center_classifier_bacc(X, center_ids, seed=42)
    assert mean_bacc > 0.50


def test_center_classifier_identical(identical_data):
    """All centers from same distribution → BACC near chance (33%)."""
    X, center_ids = identical_data
    mean_bacc, _ = center_classifier_bacc(X, center_ids, seed=42)
    assert mean_bacc < 0.55  # should be near 0.33, allow margin


# ── MMD tests ────────────────────────────────────────────────


def test_mmd_shifted_larger(shifted_data):
    """MMD(A,B) > MMD(A,C) when B is shifted."""
    X, center_ids = shifted_data
    results = mmd_pairwise(X, center_ids)
    assert results[("A", "B")] > results[("A", "C")]


def test_mmd_symmetric(shifted_data):
    """MMD(A,B) == MMD(B,A)."""
    X, center_ids = shifted_data
    results = mmd_pairwise(X, center_ids)
    assert results[("A", "B")] == results[("B", "A")]


# ── UMAP tests ───────────────────────────────────────────────


@pytest.mark.skipif(not HAS_UMAP, reason="umap-learn not installed")
def test_umap_output_shape(shifted_data):
    """Embedding shape is (n_samples, 2)."""
    X, center_ids = shifted_data
    result = umap_by_center(X, center_ids, seed=42)
    assert result["embedding"].shape == (X.shape[0], 2)
    assert len(result["center_ids"]) == X.shape[0]
