"""Tests for claryon.io.tabular — CSV/Parquet tabular loader."""
from __future__ import annotations

import numpy as np
import pandas as pd

from claryon.io.base import TaskType
from claryon.io.tabular import load_tabular_csv, _infer_task_type


def test_load_binary_csv(tabular_binary_dir):
    ds = load_tabular_csv(tabular_binary_dir / "train.csv")
    assert ds.n_samples == 80
    assert ds.n_features == 10
    assert ds.task_type == TaskType.BINARY
    assert ds.y is not None
    assert set(np.unique(ds.y)) == {0, 1}
    assert ds.label_mapper is not None
    assert len(ds.feature_names) == 10


def test_load_multiclass_csv(tabular_multiclass_dir):
    ds = load_tabular_csv(tabular_multiclass_dir / "train.csv")
    assert ds.n_samples == 120
    assert ds.task_type == TaskType.MULTICLASS
    assert set(np.unique(ds.y)) == {0, 1, 2}


def test_load_regression_csv(tabular_regression_dir):
    ds = load_tabular_csv(
        tabular_regression_dir / "train.csv",
        label_col="target",
        task_type=TaskType.REGRESSION,
    )
    assert ds.n_samples == 80
    assert ds.task_type == TaskType.REGRESSION
    assert ds.y.dtype == np.float64


def test_load_csv_no_id_col(tabular_binary_dir):
    ds = load_tabular_csv(tabular_binary_dir / "train.csv", id_col=None)
    assert ds.keys[0] == "S0000"
    assert len(ds.keys) == ds.n_samples


def test_load_csv_no_label(tmp_path):
    df = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0]})
    csv_path = tmp_path / "nolabel.csv"
    df.to_csv(csv_path, sep=";", index=False)
    ds = load_tabular_csv(csv_path, label_col="label")
    assert ds.y is None
    assert ds.n_features == 2


def test_load_csv_max_features(tabular_binary_dir):
    ds = load_tabular_csv(tabular_binary_dir / "train.csv", max_features=3)
    assert ds.n_features == 3


def test_load_csv_shapes_consistent(tabular_binary_dir):
    train = load_tabular_csv(tabular_binary_dir / "train.csv")
    test = load_tabular_csv(tabular_binary_dir / "test.csv")
    assert train.n_features == test.n_features


def test_infer_task_type_binary():
    y = pd.Series([0, 1, 0, 1])
    assert _infer_task_type(y) == TaskType.BINARY


def test_infer_task_type_multiclass():
    y = pd.Series([0, 1, 2, 0, 1, 2])
    assert _infer_task_type(y) == TaskType.MULTICLASS


def test_infer_task_type_regression():
    y = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    assert _infer_task_type(y) == TaskType.REGRESSION
