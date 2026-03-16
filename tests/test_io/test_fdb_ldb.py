"""Tests for claryon.io.fdb_ldb — FDB/LDB legacy format loader."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.io.base import TaskType
from claryon.io.fdb_ldb import load_fdb_ldb, write_fdb_ldb


def test_load_fdb_ldb(fdb_path, ldb_path):
    ds = load_fdb_ldb(fdb_path, ldb_path)
    assert ds.n_samples > 0
    assert ds.y is not None
    assert ds.label_mapper is not None
    assert ds.keys[0] == "S0000"


def test_fdb_ldb_feature_names(fdb_path, ldb_path):
    ds = load_fdb_ldb(fdb_path, ldb_path)
    assert len(ds.feature_names) == ds.n_features
    assert all(isinstance(n, str) for n in ds.feature_names)


def test_fdb_ldb_roundtrip(fdb_path, ldb_path, output_dir):
    ds = load_fdb_ldb(fdb_path, ldb_path)
    write_fdb_ldb(ds, output_dir / "FDB.csv", output_dir / "LDB.csv")
    ds2 = load_fdb_ldb(output_dir / "FDB.csv", output_dir / "LDB.csv")
    assert ds2.n_samples == ds.n_samples
    assert ds2.n_features == ds.n_features
    np.testing.assert_array_almost_equal(ds2.X, ds.X, decimal=6)


def test_fdb_ldb_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fdb_ldb(tmp_path / "nope.csv", tmp_path / "nope2.csv")
