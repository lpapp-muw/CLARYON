"""Synthetic multi-center NIfTI-like data for integration testing."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _spherical_mask(shape: Tuple[int, int, int], radius_frac: float = 0.4) -> np.ndarray:
    """Create a centered spherical binary mask.

    Args:
        shape: Volume dimensions (D, H, W).
        radius_frac: Radius as fraction of half the smallest side.

    Returns:
        Binary mask array of given shape.
    """
    d, h, w = shape
    center = np.array([d / 2, h / 2, w / 2])
    radius = radius_frac * min(d, h, w) / 2

    zz, yy, xx = np.mgrid[:d, :h, :w]
    dist = np.sqrt(
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    )
    return (dist <= radius).astype(np.float64)


def generate_multicenter_volumes(
    n_per_center: int = 20,
    shape: Tuple[int, int, int] = (16, 16, 16),
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate synthetic 3-center volumetric data with domain shift.

    Centers simulate different PET acquisition characteristics:
    - "DE": base intensity, moderate noise (PET/CT baseline)
    - "CH": 1.3x intensity scale, higher noise (different AC method)
    - "AT": 1.1x intensity + 0.5 additive offset, moderate noise (dual-tracer)

    Label 1 volumes have slightly higher mean intensity than label 0
    (simulating higher-grade tumors with more tracer uptake).

    Args:
        n_per_center: Number of samples per center.
        shape: Volume dimensions. Must be power-of-2 cube for Hilbert.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'volumes', 'masks', 'labels', 'center_ids', 'keys'.
    """
    rng = np.random.default_rng(seed)
    n_total = 3 * n_per_center
    mask = _spherical_mask(shape)

    volumes = np.zeros((n_total, *shape), dtype=np.float64)
    masks = np.tile(mask, (n_total, 1, 1, 1))
    labels = np.zeros(n_total, dtype=np.int64)
    center_ids: List[str] = []
    keys: List[str] = []

    # Balanced binary labels per center
    labels_per_center = np.array([0, 1] * ((n_per_center + 1) // 2))[:n_per_center]

    center_configs = {
        "DE": {"scale": 1.0, "offset": 0.0, "noise_std": 0.10},
        "CH": {"scale": 1.3, "offset": 0.0, "noise_std": 0.15},
        "AT": {"scale": 1.1, "offset": 0.5, "noise_std": 0.12},
    }

    idx = 0
    for center_name, cfg in center_configs.items():
        for j in range(n_per_center):
            label = labels_per_center[j]

            # Base intensity: label 1 has higher mean (tumor aggressiveness)
            base_intensity = 1.0 + 0.3 * label
            vol = rng.normal(base_intensity, 0.2, shape)
            vol = np.abs(vol)  # PET values are non-negative

            # Apply center-specific domain shift
            vol = vol * cfg["scale"] + cfg["offset"]

            # Add center-specific noise
            vol += rng.normal(0, cfg["noise_std"], shape)
            vol = np.maximum(vol, 0.0)  # clip to non-negative

            # Apply mask (zero outside)
            vol = vol * mask

            volumes[idx] = vol
            labels[idx] = label
            center_ids.append(center_name)
            keys.append(f"{center_name}_{j + 1:03d}")
            idx += 1

    return {
        "volumes": volumes,
        "masks": masks,
        "labels": labels,
        "center_ids": np.array(center_ids),
        "keys": keys,
    }
