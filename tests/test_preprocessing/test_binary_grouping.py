"""Tests for binary grouping — multiclass to binary relabeling."""
from __future__ import annotations

import numpy as np

from claryon.config_schema import BinaryGroupingConfig
from claryon.preprocessing.binary_grouping import apply_binary_grouping


def test_4class_to_binary():
    """Group [2, 3] as positive, [0, 1] as negative."""
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    cfg = BinaryGroupingConfig(enabled=True, positive=[2, 3], negative=[0, 1])

    y_new = apply_binary_grouping(y, cfg)

    expected = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    np.testing.assert_array_equal(y_new, expected)


def test_positive_only_rest_negative():
    """If negative is empty, everything NOT in positive becomes negative."""
    y = np.array([0, 1, 2, 3])
    cfg = BinaryGroupingConfig(enabled=True, positive=[3])

    y_new = apply_binary_grouping(y, cfg)

    expected = np.array([0, 0, 0, 1])
    np.testing.assert_array_equal(y_new, expected)


def test_disabled_returns_unchanged():
    """When not enabled, labels should pass through unchanged."""
    y = np.array([0, 1, 2])
    cfg = BinaryGroupingConfig(enabled=False)

    y_new = apply_binary_grouping(y, cfg)

    np.testing.assert_array_equal(y_new, y)
