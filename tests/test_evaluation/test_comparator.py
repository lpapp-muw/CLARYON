"""Tests for claryon.evaluation.comparator — Friedman/Nemenyi, bootstrap CI."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.evaluation.comparator import (
    bootstrap_ci,
    compute_ranks,
    friedman_nemenyi,
)


def test_compute_ranks_simple():
    # 3 datasets, 3 methods. Higher is better.
    perf = np.array([
        [0.9, 0.8, 0.7],  # method 0 best
        [0.7, 0.9, 0.8],  # method 1 best
        [0.8, 0.7, 0.9],  # method 2 best
    ])
    ranks = compute_ranks(perf, higher_is_better=True)
    # Each method should be rank 1 in one dataset
    assert ranks[0, 0] == 1.0
    assert ranks[1, 1] == 1.0
    assert ranks[2, 2] == 1.0


def test_friedman_nemenyi_equal():
    # Equal performance → high p-value
    ranks = np.ones((10, 3)) * 2.0  # all tied
    result = friedman_nemenyi(ranks)
    assert result["p_value"] > 0.05
    assert not result["significant"]


def test_friedman_nemenyi_significant():
    # Method 0 always rank 1, method 2 always rank 3
    ranks = np.tile([1, 2, 3], (20, 1)).astype(float)
    result = friedman_nemenyi(ranks)
    assert result["p_value"] < 0.05
    assert result["significant"]
    assert result["critical_difference"] > 0


def test_bootstrap_ci():
    values = np.array([0.8, 0.85, 0.82, 0.83, 0.81])
    mean, lo, hi = bootstrap_ci(values, confidence=0.95, seed=42)
    assert lo < mean < hi
    assert abs(mean - values.mean()) < 1e-10
    assert hi - lo > 0
