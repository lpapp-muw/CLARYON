"""Tests for image normalization."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.preprocessing.image_prep import normalize_images


def test_per_image_range():
    """Per-image mode should produce [0, 1] range per volume."""
    volumes = np.array([
        [[0.0, 10.0], [5.0, 20.0]],
        [[100.0, 200.0], [150.0, 300.0]],
    ])
    result, _, _ = normalize_images(volumes, mode="per_image")

    for i in range(result.shape[0]):
        assert result[i].min() == pytest.approx(0.0)
        assert result[i].max() == pytest.approx(1.0)


def test_cohort_global_uses_training_bounds():
    """Cohort-global mode should use provided min/max from training set."""
    volumes = np.array([[[50.0, 100.0]]])
    result, used_min, used_max = normalize_images(
        volumes, mode="cohort_global", global_min=0.0, global_max=200.0,
    )

    np.testing.assert_almost_equal(result[0, 0, 0], 0.25)
    np.testing.assert_almost_equal(result[0, 0, 1], 0.5)
    assert used_min == 0.0
    assert used_max == 200.0


def test_cohort_global_clips_out_of_range():
    """Values outside training range should be clipped to [0, 1]."""
    volumes = np.array([[[-10.0, 250.0]]])
    result, _, _ = normalize_images(
        volumes, mode="cohort_global", global_min=0.0, global_max=200.0,
    )
    np.testing.assert_almost_equal(result[0, 0, 0], 0.0)
    np.testing.assert_almost_equal(result[0, 0, 1], 1.0)
