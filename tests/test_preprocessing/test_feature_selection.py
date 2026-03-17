"""Tests for mRMR feature selection."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.preprocessing.feature_selection import mrmr_select


def test_perfectly_correlated_features():
    """Two perfectly correlated features — the one more relevant to label is kept."""
    rng = np.random.default_rng(42)
    n = 100
    # Feature 0: random noise (low relevance)
    f0 = rng.standard_normal(n)
    # Feature 1: correlated with label (high relevance)
    y = (rng.standard_normal(n) > 0).astype(float)
    f1 = y * 2 + rng.standard_normal(n) * 0.1
    # Feature 2: perfect copy of f1 (redundant)
    f2 = f1.copy()
    # Feature 3: another independent feature
    f3 = rng.standard_normal(n)
    # Feature 4: independent
    f4 = rng.standard_normal(n)

    X = np.column_stack([f0, f1, f2, f3, f4])
    names = ["f0", "f1", "f2", "f3", "f4"]

    selected_idx, selected_names = mrmr_select(X, y, names, spearman_threshold=0.8)

    # f1 and f2 are redundant; the one with higher relevance to y should be kept
    # Only one of f1/f2 should appear
    assert not (1 in selected_idx and 2 in selected_idx), "Both f1 and f2 kept — redundancy not removed"
    assert 1 in selected_idx or 2 in selected_idx, "Neither f1 nor f2 kept"


def test_threshold_1_keeps_everything():
    """threshold=1.0 means no features are considered redundant → keep all."""
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, 6))
    y = rng.choice([0, 1], size=n)
    names = [f"f{i}" for i in range(6)]

    selected_idx, selected_names = mrmr_select(X, y, names, spearman_threshold=1.0)

    assert len(selected_idx) == 6


def test_threshold_0_reduces_heavily():
    """threshold=0.0 means almost everything is redundant → very few features."""
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, 10))
    y = rng.choice([0, 1], size=n)
    names = [f"f{i}" for i in range(10)]

    selected_idx, selected_names = mrmr_select(X, y, names, spearman_threshold=0.0)

    # With threshold=0, any non-zero correlation clusters features
    # Should get very few features (likely 1)
    assert len(selected_idx) < 10


def test_guard_skips_if_few_features():
    """If n_features <= 4, mRMR should be skipped."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 3))
    y = rng.choice([0, 1], size=20)
    names = ["a", "b", "c"]

    selected_idx, selected_names = mrmr_select(X, y, names, spearman_threshold=0.8)

    assert selected_idx == [0, 1, 2]
    assert selected_names == ["a", "b", "c"]


def test_max_features_cap():
    """max_features should cap the number of selected features."""
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, 10))
    y = rng.choice([0, 1], size=n)
    names = [f"f{i}" for i in range(10)]

    selected_idx, _ = mrmr_select(X, y, names, spearman_threshold=1.0, max_features=3)

    assert len(selected_idx) <= 3
