"""Tests for bootstrap_metric_ci, paired_superiority, and mcnemar_test."""
from __future__ import annotations

import numpy as np

from sklearn.metrics import balanced_accuracy_score

from claryon.evaluation.comparator import (
    bootstrap_metric_ci,
    mcnemar_test,
    paired_superiority,
)


def _bacc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return balanced_accuracy_score(y_true, y_pred)


# ── bootstrap_metric_ci tests ───────────────────────────────


def test_bootstrap_metric_ci_basic():
    """CI brackets the point estimate."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5)
    y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1] * 5)
    mean, lo, hi = bootstrap_metric_ci(y_true, y_pred, _bacc)
    assert 0.7 < lo < mean < hi < 1.0


def test_bootstrap_metric_ci_perfect():
    """Perfect predictions → CI near 1.0."""
    y = np.array([0, 1, 0, 1, 0, 1] * 10)
    mean, lo, hi = bootstrap_metric_ci(y, y, _bacc)
    assert mean == 1.0
    assert lo > 0.99


def test_bootstrap_metric_ci_reproducible():
    """Same seed → identical results."""
    y_true = np.array([0, 1] * 25)
    y_pred = np.array([0, 0] * 25)
    r1 = bootstrap_metric_ci(y_true, y_pred, _bacc, seed=42)
    r2 = bootstrap_metric_ci(y_true, y_pred, _bacc, seed=42)
    assert r1 == r2


# ── paired_superiority tests ────────────────────────────────


def test_paired_superiority_clearly_better():
    """Perfect A vs always-0 B → P(A>B) near 1.0."""
    y_true = np.array([0, 1, 0, 1] * 20)
    preds_good = y_true.copy()
    preds_bad = np.zeros_like(y_true)
    p = paired_superiority(y_true, preds_good, preds_bad, _bacc)
    assert p > 0.95


def test_paired_superiority_equal():
    """Equally good but different predictions → P(A>B) near 0.5."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    # Both get ~80% right but on different samples
    preds_a = y_true.copy()
    preds_b = y_true.copy()
    flip_a = rng.choice(100, size=20, replace=False)
    flip_b = rng.choice(100, size=20, replace=False)
    preds_a[flip_a] = 1 - preds_a[flip_a]
    preds_b[flip_b] = 1 - preds_b[flip_b]
    p = paired_superiority(y_true, preds_a, preds_b, _bacc)
    assert 0.2 < p < 0.8


# ── mcnemar_test tests ──────────────────────────────────────


def test_mcnemar_identical():
    """Same predictions → not significant."""
    y_true = np.array([0, 1, 0, 1] * 20)
    preds = np.array([0, 1, 0, 0] * 20)
    stat, pval = mcnemar_test(y_true, preds, preds)
    assert pval > 0.05


def test_mcnemar_different():
    """One always right, other always wrong → highly significant."""
    y_true = np.array([0, 1, 0, 1] * 20)
    preds_a = y_true.copy()
    preds_b = 1 - y_true
    stat, pval = mcnemar_test(y_true, preds_a, preds_b)
    assert pval < 0.01
