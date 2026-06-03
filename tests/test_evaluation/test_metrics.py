"""Tests for claryon.evaluation.metrics — metric implementations."""
from __future__ import annotations

import numpy as np

from claryon.evaluation.metrics import (
    binary_metrics,
    metric_accuracy,
    metric_bacc,
    metric_sensitivity,
    metric_specificity,
    metric_ppv,
    metric_npv,
    metric_auc,
    metric_mse,
    metric_mae,
    metric_r2,
    select_threshold_balanced_accuracy,
)


def test_binary_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    p1 = np.array([0.1, 0.2, 0.8, 0.9])
    m = binary_metrics(y, p1, threshold=0.5)
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["tp"] == 2 and m["tn"] == 2


def test_binary_metrics_worst():
    y = np.array([0, 0, 1, 1])
    p1 = np.array([0.9, 0.8, 0.1, 0.2])
    m = binary_metrics(y, p1, threshold=0.5)
    assert m["sensitivity"] == 0.0
    assert m["specificity"] == 0.0


def test_metric_accuracy():
    assert metric_accuracy(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])) == 1.0
    assert metric_accuracy(np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0])) == 0.0


def test_metric_bacc():
    assert metric_bacc(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1.0


def test_metric_sensitivity():
    assert metric_sensitivity(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1.0


def test_metric_specificity():
    assert metric_specificity(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1.0


def test_metric_ppv():
    assert metric_ppv(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1.0


def test_metric_npv():
    assert metric_npv(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1.0


def test_metric_auc():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    v = metric_auc(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), probabilities=probs)
    assert v == 1.0


def test_metric_mse():
    assert metric_mse(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0


def test_metric_mae():
    assert metric_mae(np.array([1.0, 3.0]), np.array([1.0, 3.0])) == 0.0


def test_metric_r2():
    assert metric_r2(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 1.0


def test_threshold_selection():
    y = np.array([0, 0, 0, 1, 1, 1])
    p1 = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = select_threshold_balanced_accuracy(y, p1)
    assert 0.0 < thr < 1.0


def test_metrics_registered():
    from claryon.registry import get
    assert callable(get("metric", "bacc"))
    assert callable(get("metric", "accuracy"))
    assert callable(get("metric", "auc"))
    assert callable(get("metric", "mse"))
