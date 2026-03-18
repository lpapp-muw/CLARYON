"""Tests for claryon.preprocessing.tabular_prep — imputation, encoding, normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd

from claryon.preprocessing.tabular_prep import (
    detect_categorical_columns,
    preprocess_tabular,
)


def test_detect_categorical_columns():
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "gender": ["M", "F", "M"],
        "score": [0.5, 0.8, 0.3],
    })
    cats = detect_categorical_columns(df)
    assert "gender" in cats
    assert "age" not in cats
    assert "score" not in cats


def test_detect_with_known_categoricals():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [4, 5, 6],
    })
    cats = detect_categorical_columns(df, known_categoricals=["x"])
    assert "x" in cats


def test_preprocess_numeric_only():
    df = pd.DataFrame({
        "f0": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        "f1": [5.0, 4.0, 3.0, 2.0, 1.0] * 10,
    })
    result = preprocess_tabular(df, use_quantile=True)
    assert result.X.shape == (50, 2)
    assert len(result.feature_names) == 2
    assert not np.any(np.isnan(result.X))


def test_preprocess_with_missing():
    df = pd.DataFrame({
        "f0": [1.0, np.nan, 3.0, 4.0],
        "f1": [5.0, 6.0, 7.0, 8.0],
    })
    result = preprocess_tabular(df, use_quantile=False, missing_indicator=True)
    assert "f0_missing" in result.feature_names
    assert result.X.shape[1] == 3  # f0, f1, f0_missing
    # Missing indicator should be 1 for row 1
    idx = result.feature_names.index("f0_missing")
    assert result.X[1, idx] == 1.0
    assert result.X[0, idx] == 0.0


def test_preprocess_with_categoricals():
    df = pd.DataFrame({
        "color": ["red", "blue", "red", "green"] * 10,
        "value": [1.0, 2.0, 3.0, 4.0] * 10,
    })
    result = preprocess_tabular(df, categorical_columns=["color"], use_quantile=True)
    # Should have value + one-hot columns for color
    assert result.X.shape[0] == 40
    assert any("color=" in n for n in result.feature_names)
    assert "value" in result.feature_names


def test_preprocess_no_quantile():
    df = pd.DataFrame({
        "f0": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    result = preprocess_tabular(df, use_quantile=False)
    # Should be raw values
    np.testing.assert_array_almost_equal(result.X[:, 0], [1.0, 2.0, 3.0, 4.0, 5.0])


def test_preprocess_metadata():
    df = pd.DataFrame({
        "f0": [1.0, np.nan, 3.0] * 20,
        "cat": ["a", "b", "a"] * 20,
    })
    result = preprocess_tabular(df, categorical_columns=["cat"], use_quantile=True)
    assert "imputation" in result.metadata
    assert "encoding" in result.metadata
    assert "normalization" in result.metadata
