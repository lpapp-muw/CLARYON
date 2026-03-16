"""Tests for claryon.io.predictions — write/read round-trip, format compliance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from claryon.io.base import TaskType
from claryon.io.predictions import infer_task_type, read_predictions, write_predictions


@pytest.fixture
def binary_data():
    keys = [f"S{i:04d}" for i in range(5)]
    actual = np.array([0, 1, 0, 1, 0])
    probs = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.4, 0.6],
        [0.85, 0.15],
    ])
    predicted = np.argmax(probs, axis=1)
    return keys, actual, predicted, probs


def test_binary_roundtrip(tmp_path, binary_data):
    keys, actual, predicted, probs = binary_data
    path = tmp_path / "pred.csv"

    write_predictions(
        path, keys, actual, predicted, probs,
        task_type=TaskType.BINARY, threshold=0.5, fold=0, seed=42,
    )

    df = read_predictions(path)
    assert list(df.columns[:5]) == ["Key", "Actual", "Predicted", "P0", "P1"]
    assert len(df) == 5
    assert df["Key"].tolist() == keys
    assert df["Actual"].tolist() == list(actual)
    assert df["Predicted"].tolist() == list(predicted)

    # Verify separator is semicolon
    raw = path.read_text()
    assert ";" in raw.split("\n")[0]
    assert "," not in raw.split("\n")[0]

    # Verify float precision (8 decimal places)
    assert "0.90000000" in raw
    assert "0.10000000" in raw


def test_multiclass_roundtrip(tmp_path):
    keys = ["A", "B", "C"]
    actual = np.array([0, 1, 2])
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7],
    ])
    predicted = np.argmax(probs, axis=1)

    path = tmp_path / "mc.csv"
    write_predictions(path, keys, actual, predicted, probs, task_type=TaskType.MULTICLASS)

    df = read_predictions(path)
    assert "P0" in df.columns
    assert "P1" in df.columns
    assert "P2" in df.columns
    assert infer_task_type(df) == TaskType.MULTICLASS


def test_regression_roundtrip(tmp_path):
    keys = ["X0", "X1", "X2"]
    actual = np.array([1.5, 2.3, 3.7])
    predicted = np.array([1.4, 2.5, 3.6])

    path = tmp_path / "reg.csv"
    write_predictions(path, keys, actual, predicted, task_type=TaskType.REGRESSION)

    df = read_predictions(path)
    assert list(df.columns[:3]) == ["Key", "Actual", "Predicted"]
    assert "P0" not in df.columns
    assert infer_task_type(df) == TaskType.REGRESSION


def test_inference_only_no_actual(tmp_path):
    keys = ["P1", "P2"]
    predicted = np.array([0, 1])
    probs = np.array([[0.6, 0.4], [0.3, 0.7]])

    path = tmp_path / "infer.csv"
    write_predictions(path, keys, None, predicted, probs, task_type=TaskType.BINARY)

    df = read_predictions(path)
    # Actual column should be empty strings → NaN after read
    assert df["Actual"].isna().all() or (df["Actual"] == "").all()


def test_missing_probs_classification_raises(tmp_path):
    with pytest.raises(ValueError, match="probabilities required"):
        write_predictions(
            tmp_path / "bad.csv",
            ["A"], np.array([0]), np.array([0]),
            probabilities=None, task_type=TaskType.BINARY,
        )


def test_shape_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="keys has"):
        write_predictions(
            tmp_path / "bad.csv",
            ["A", "B"], np.array([0]), np.array([0, 1]),
            probabilities=np.array([[0.5, 0.5], [0.5, 0.5]]),
            task_type=TaskType.BINARY,
        )
