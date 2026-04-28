"""3D Hilbert curve flattening for volumetric data."""
from __future__ import annotations

import logging
import math
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# Module-level cache for computed indices
_HILBERT_CACHE: Dict[int, np.ndarray] = {}


def hilbert_3d_indices(side_length: int) -> np.ndarray:
    """Compute 3D Hilbert curve index mapping for a cubic volume.

    Returns a 1D array where element i is the row-major flattened index
    of the voxel visited at Hilbert step i. The result is cached for
    repeated calls with the same side_length.

    Args:
        side_length: Side length of the cubic volume. Must be a power of 2.

    Returns:
        1D array of length side_length³. Element i is the flattened
        (row-major) index of the voxel visited at Hilbert step i.

    Raises:
        ValueError: If side_length is not a power of 2 or is less than 2.
    """
    if side_length < 2 or (side_length & (side_length - 1)) != 0:
        raise ValueError(
            f"side_length must be a power of 2 (>= 2), got {side_length}"
        )

    if side_length in _HILBERT_CACHE:
        return _HILBERT_CACHE[side_length]

    try:
        from hilbertcurve.hilbertcurve import HilbertCurve
    except ImportError:
        raise ImportError(
            "hilbertcurve package is required for Hilbert flattening. "
            "Install with: pip install hilbertcurve"
        ) from None

    p = int(math.log2(side_length))
    hc = HilbertCurve(p, 3)
    n_voxels = side_length ** 3
    side2 = side_length * side_length

    indices = np.empty(n_voxels, dtype=np.int64)
    for d in range(n_voxels):
        x, y, z = hc.point_from_distance(d)
        indices[d] = x * side2 + y * side_length + z

    _HILBERT_CACHE[side_length] = indices
    logger.info(
        "Computed 3D Hilbert curve for side=%d (%d voxels, p=%d)",
        side_length, n_voxels, p,
    )
    return indices


def flatten_volume(volume: np.ndarray, order: str = "rowmajor") -> np.ndarray:
    """Flatten a 3D volume to 1D with specified ordering.

    Args:
        volume: 3D array (D, H, W). Must be cubic for Hilbert ordering.
        order: ``"rowmajor"`` (default, equivalent to ravel) or ``"hilbert"``.

    Returns:
        1D float64 array of length D*H*W.

    Raises:
        ValueError: If order is ``"hilbert"`` and volume is not cubic
            or side_length is not a power of 2.
        ValueError: If order is unknown.
    """
    if order == "rowmajor":
        return volume.ravel().astype(np.float64)

    if order == "hilbert":
        if volume.ndim != 3:
            raise ValueError(
                f"Hilbert flattening requires a 3D volume, got {volume.ndim}D"
            )
        d, h, w = volume.shape
        if not (d == h == w):
            raise ValueError(
                f"Hilbert flattening requires a cubic volume (D==H==W), "
                f"got shape ({d}, {h}, {w})"
            )
        indices = hilbert_3d_indices(d)
        flat = volume.ravel().astype(np.float64)
        return flat[indices]

    raise ValueError(f"Unknown flatten order: {order!r}. Use 'rowmajor' or 'hilbert'.")
