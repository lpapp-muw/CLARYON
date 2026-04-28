"""Tests for claryon.io.hilbert — Hilbert curve 3D flattening."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.io.hilbert import flatten_volume, hilbert_3d_indices


def test_hilbert_is_permutation():
    """Hilbert indices are a permutation of [0, side³)."""
    idx = hilbert_3d_indices(8)
    assert len(idx) == 512
    assert set(idx.tolist()) == set(range(512))


def test_hilbert_preserves_values():
    """Hilbert and rowmajor have identical value sets and sum."""
    vol = np.random.default_rng(42).random((8, 8, 8))
    flat_rm = flatten_volume(vol, "rowmajor")
    flat_hb = flatten_volume(vol, "hilbert")
    assert np.isclose(flat_rm.sum(), flat_hb.sum())
    assert set(np.round(flat_rm, 10)) == set(np.round(flat_hb, 10))


def test_hilbert_locality():
    """Median 1D neighbor distance is lower for Hilbert than rowmajor.

    Uses 32×32×32 (the actual prostate VOI size). The Hilbert curve
    dramatically reduces the typical distance between 3D face-neighbors
    in the 1D ordering (median 5 vs 32 for rowmajor), even though the
    mean can be similar due to a long tail of outlier distances.
    """
    side = 32
    idx = hilbert_3d_indices(side)

    # Build reverse map: rowmajor_index → hilbert_position
    reverse_hilbert = np.empty(side ** 3, dtype=np.int64)
    reverse_hilbert[idx] = np.arange(side ** 3)

    # For each voxel, check 6 face-neighbors
    hilbert_dists = []
    rowmajor_dists = []
    offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    for x in range(side):
        for y in range(side):
            for z in range(side):
                rm_idx = x * side * side + y * side + z
                for dx, dy, dz in offsets:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < side and 0 <= ny < side and 0 <= nz < side:
                        rm_neighbor = nx * side * side + ny * side + nz
                        # Rowmajor distance
                        rowmajor_dists.append(abs(rm_idx - rm_neighbor))
                        # Hilbert distance
                        hilbert_dists.append(
                            abs(int(reverse_hilbert[rm_idx]) - int(reverse_hilbert[rm_neighbor]))
                        )

    # Median captures typical neighbor distance; Hilbert is much better
    assert np.median(hilbert_dists) < np.median(rowmajor_dists)
    # 75th percentile also dramatically lower
    assert np.percentile(hilbert_dists, 75) < np.percentile(rowmajor_dists, 75)


def test_hilbert_requires_cubic():
    """Non-cubic input raises ValueError."""
    vol = np.zeros((8, 8, 16))
    with pytest.raises(ValueError, match="cubic"):
        flatten_volume(vol, "hilbert")


def test_hilbert_requires_power_of_2():
    """Non-power-of-2 side length raises ValueError."""
    vol = np.zeros((6, 6, 6))
    with pytest.raises(ValueError, match="power of 2"):
        flatten_volume(vol, "hilbert")


def test_flatten_rowmajor_matches_ravel():
    """rowmajor ordering matches numpy ravel."""
    vol = np.random.default_rng(42).random((8, 8, 8))
    np.testing.assert_array_equal(flatten_volume(vol, "rowmajor"), vol.ravel())


def test_flatten_unknown_order():
    """Unknown order raises ValueError."""
    with pytest.raises(ValueError, match="Unknown"):
        flatten_volume(np.zeros((8, 8, 8)), "invalid")


def test_hilbert_32_cube():
    """Works at actual prostate VOI size (32³ = 32768 voxels)."""
    idx = hilbert_3d_indices(32)
    assert len(idx) == 32768
    assert set(idx.tolist()) == set(range(32768))


def test_hilbert_16_cube():
    """Works at intermediate size (16³ = 4096 voxels)."""
    idx = hilbert_3d_indices(16)
    assert len(idx) == 4096
    assert set(idx.tolist()) == set(range(4096))
