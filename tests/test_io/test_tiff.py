"""Tests for claryon.io.tiff — TIFF image loader."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.io.base import TaskType


def test_load_tiff_dataset(tiff_dir):
    pytest.importorskip("tifffile")
    from claryon.io.tiff import load_tiff_dataset

    ds = load_tiff_dataset(tiff_dir)
    assert ds.n_samples == 5
    assert ds.y is not None
    assert ds.task_type == TaskType.BINARY
    assert set(np.unique(ds.y)) == {0, 1}


def test_tiff_keys_from_sidecar(tiff_dir):
    pytest.importorskip("tifffile")
    from claryon.io.tiff import load_tiff_dataset

    ds = load_tiff_dataset(tiff_dir)
    # Keys should be from the stem (since JSON sidecars don't have "id")
    assert all(k.startswith("sample_") for k in ds.keys)


def test_tiff_no_files_raises(tmp_path):
    from claryon.io.tiff import load_tiff_dataset

    with pytest.raises(FileNotFoundError, match="No TIFF files"):
        load_tiff_dataset(tmp_path)
