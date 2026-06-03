"""Tests for claryon.models.ensemble — prediction aggregation."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.io.base import TaskType
from claryon.models.ensemble import ensemble_predictions


def test_binary_ensemble():
    # 3 models, 5 samples, 2 classes
    p1 = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9], [0.5, 0.5]])
    p2 = np.array([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5], [0.2, 0.8], [0.6, 0.4]])
    p3 = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7], [0.4, 0.6]])

    preds, avg_probs = ensemble_predictions([p1, p2, p3], TaskType.BINARY)
    assert preds.shape == (5,)
    assert avg_probs.shape == (5, 2)
    # First sample: avg P0 = (0.8+0.7+0.9)/3 = 0.8 > 0.5 → class 0
    assert preds[0] == 0
    # Second sample: avg P1 = (0.7+0.6+0.8)/3 = 0.7 > 0.5 → class 1
    assert preds[1] == 1


def test_multiclass_ensemble():
    p1 = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    p2 = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    preds, avg_probs = ensemble_predictions([p1, p2], TaskType.MULTICLASS)
    assert preds.shape == (2,)
    assert avg_probs.shape == (2, 3)
    assert preds[0] == 0  # class 0 highest
    assert preds[1] == 2  # class 2 highest


def test_regression_ensemble():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([1.5, 2.5, 3.5])
    v3 = np.array([1.2, 2.2, 3.2])
    preds, avg = ensemble_predictions([v1, v2, v3], TaskType.REGRESSION)
    np.testing.assert_almost_equal(preds, [1.2333, 2.2333, 3.2333], decimal=3)


def test_empty_raises():
    with pytest.raises(ValueError, match="No prediction"):
        ensemble_predictions([], TaskType.BINARY)
