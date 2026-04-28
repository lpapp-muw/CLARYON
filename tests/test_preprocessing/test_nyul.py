"""Tests for Nyúl histogram matching — nyul_fit() and nyul_transform()."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.preprocessing.image_prep import nyul_fit, nyul_transform

PERCENTILES = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99)


@pytest.fixture
def center_volumes():
    """Synthetic volumes for 3 centers with different intensity distributions.

    A: uniform(0.1, 1.0), B: uniform(0.6, 2.5) (shifted+scaled), C: uniform(0.1, 1.0).
    """
    rng = np.random.default_rng(42)
    n = 5
    shape = (8, 8, 8)
    vols_a = [rng.uniform(0.1, 1.0, shape) for _ in range(n)]
    vols_b = [rng.uniform(0.6, 2.5, shape) for _ in range(n)]
    vols_c = [rng.uniform(0.1, 1.0, shape) for _ in range(n)]
    return vols_a, vols_b, vols_c


def test_nyul_fit_shape(center_volumes):
    """Reference landmarks shape matches number of percentiles."""
    vols_a, _, vols_c = center_volumes
    landmarks = nyul_fit(vols_a + vols_c, percentiles=PERCENTILES)
    assert landmarks.shape == (len(PERCENTILES),)


def test_nyul_transform_percentiles_align(center_volumes):
    """After transform, B's percentiles should be close to reference."""
    vols_a, vols_b, _ = center_volumes
    landmarks = nyul_fit(vols_a, percentiles=PERCENTILES)
    b_transformed = nyul_transform(vols_b[0], landmarks, percentiles=PERCENTILES)
    fg = b_transformed[b_transformed > 0]
    b_percs = np.percentile(fg, [10, 50, 90])
    ref_percs = landmarks[[1, 5, 9]]  # indices for 10th, 50th, 90th
    np.testing.assert_allclose(b_percs, ref_percs, atol=0.15)


def test_nyul_no_leakage(center_volumes):
    """Fit on train only vs fit on all should differ."""
    vols_a, vols_b, _ = center_volumes
    landmarks_train = nyul_fit(vols_a, percentiles=PERCENTILES)
    landmarks_all = nyul_fit(vols_a + vols_b, percentiles=PERCENTILES)
    assert not np.allclose(landmarks_train, landmarks_all)


def test_nyul_masked_voxels():
    """Only masked voxels are transformed. Unmasked stay at 0."""
    rng = np.random.default_rng(42)
    vol = rng.uniform(0.1, 1.0, (8, 8, 8))
    mask = np.zeros((8, 8, 8))
    mask[2:6, 2:6, 2:6] = 1

    # Apply mask to volume (zero out unmasked)
    masked_vol = vol * mask
    landmarks = nyul_fit([rng.uniform(0.2, 2.0, (8, 8, 8)) for _ in range(3)])
    out = nyul_transform(masked_vol, landmarks, mask=mask)

    assert np.all(out[mask == 0] == 0)
    # Masked voxels should have changed
    assert not np.allclose(out[mask == 1], masked_vol[mask == 1])


def test_nyul_monotonic():
    """Piecewise-linear mapping preserves ordering of voxel intensities."""
    rng = np.random.default_rng(42)
    vol = rng.uniform(0.1, 1.0, (8, 8, 8))
    landmarks = nyul_fit([rng.uniform(0.5, 3.0, (8, 8, 8)) for _ in range(3)])
    out = nyul_transform(vol, landmarks)

    fg_in = vol[vol > 0]
    fg_out = out[vol > 0]
    # For any pair where input[i] < input[j], output[i] <= output[j]
    order = np.argsort(fg_in)
    sorted_out = fg_out[order]
    assert np.all(np.diff(sorted_out) >= -1e-12)


def test_nyul_identity():
    """If volume already matches reference, output ≈ input.

    Values at the tails (below 1st or above 99th percentile) are clipped
    to the landmark boundaries — this is correct Nyúl behavior. We check
    that the vast majority of voxels are unchanged and the max error is
    small (only affects tail voxels).
    """
    rng = np.random.default_rng(42)
    vol = rng.uniform(0.1, 1.0, (8, 8, 8))
    landmarks = nyul_fit([vol])
    out = nyul_transform(vol, landmarks)
    fg = vol > 0
    # Almost all voxels should be very close; tail clipping causes small diffs
    np.testing.assert_allclose(out[fg], vol[fg], atol=0.02)


def test_nyul_does_not_modify_input():
    """Input volume must not be modified in place."""
    rng = np.random.default_rng(42)
    vol = rng.uniform(0.1, 1.0, (8, 8, 8))
    vol_copy = vol.copy()
    landmarks = nyul_fit([rng.uniform(0.5, 3.0, (8, 8, 8)) for _ in range(3)])
    _ = nyul_transform(vol, landmarks)
    np.testing.assert_array_equal(vol, vol_copy)
