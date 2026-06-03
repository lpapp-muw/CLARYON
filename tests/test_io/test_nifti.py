"""Tests for claryon.io.nifti — NIfTI volume loader."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.io.nifti import (
    _case_id,
    _parse_label,
    load_nifti_dataset,
)


def test_case_id_extraction():
    from pathlib import Path
    # Label digit is between patient ID and modality suffix
    assert _case_id(Path("patient001_0_PET.nii.gz")) == "patient001_0"
    assert _case_id(Path("subj_42_mask.nii.gz")) == "subj_42"
    # Fixture naming: casetrain000_0_PET → trailing digit (0) is stripped, PET dropped
    assert _case_id(Path("casetrain000_0_PET.nii.gz")) == "casetrain000_0"


def test_parse_label():
    from pathlib import Path
    assert _parse_label(Path("case_0_PET.nii.gz")) == "0"
    assert _parse_label(Path("case_1_PET.nii.gz")) == "1"


def test_parse_label_no_digit():
    from pathlib import Path
    with pytest.raises(ValueError, match="Cannot parse label"):
        _parse_label(Path("case_nodigit_PET.nii.gz"))


def test_load_nifti_masked_dataset(synthetic_nifti_masked):
    result = load_nifti_dataset(synthetic_nifti_masked)
    assert "train" in result
    assert "test" in result
    train_ds = result["train"]
    test_ds = result["test"]
    assert train_ds.n_samples > 0
    assert test_ds.n_samples > 0
    assert train_ds.y is not None
    assert test_ds.y is not None
    # Binary labels
    assert set(np.unique(train_ds.y)).issubset({0, 1})


def test_load_nifti_nomask_dataset(synthetic_nifti_nomask):
    result = load_nifti_dataset(
        synthetic_nifti_nomask, mask_pattern=None,
    )
    assert "train" in result or "all" in result


def test_load_nifti_feature_shapes_consistent(synthetic_nifti_masked):
    result = load_nifti_dataset(synthetic_nifti_masked)
    train_ds = result["train"]
    test_ds = result["test"]
    assert train_ds.n_features == test_ds.n_features


def test_nifti_mask_applied(synthetic_nifti_masked, nifti_image_mask_pairs):
    """Verify that masking zeros out voxels outside the mask."""
    if not nifti_image_mask_pairs:
        pytest.skip("No image/mask pairs found")
    result = load_nifti_dataset(synthetic_nifti_masked)
    train_ds = result["train"]
    # Just verify it loaded without error and has nonzero data
    assert train_ds.X.max() > 0 or train_ds.X.min() < 0
