"""Tests for z-score normalization — verify train-only fitting."""
from __future__ import annotations

import numpy as np

from claryon.preprocessing.tabular_prep import apply_zscore, fit_zscore


def test_fit_apply_zscore():
    """Z-score params fitted on train must be applied to test (not re-fitted)."""
    X_train = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    X_test = np.array([[2.0, 15.0], [6.0, 35.0]])

    mean, std = fit_zscore(X_train)

    X_train_z = apply_zscore(X_train, mean, std)
    X_test_z = apply_zscore(X_test, mean, std)

    # Train should have ~0 mean
    np.testing.assert_almost_equal(X_train_z.mean(axis=0), [0.0, 0.0], decimal=10)

    # Test should NOT have 0 mean (uses train params)
    assert not np.allclose(X_test_z.mean(axis=0), 0.0)

    # Verify manual computation
    expected_mean = np.array([3.0, 20.0])
    expected_std = np.std(X_train, axis=0)
    np.testing.assert_array_almost_equal(mean, expected_mean)
    np.testing.assert_array_almost_equal(std, expected_std)


def test_zero_std_protection():
    """Constant features should not cause division by zero."""
    X_train = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    mean, std = fit_zscore(X_train)
    result = apply_zscore(X_train, mean, std)
    assert np.all(np.isfinite(result))
