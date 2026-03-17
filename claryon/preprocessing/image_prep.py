"""Image preprocessing — resampling, normalization, augmentation basics.

New module. Basic image preprocessing utilities for NIfTI/TIFF data.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def normalize_images(
    volumes: np.ndarray,
    mode: str = "per_image",
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """Normalize image volumes to [0, 1].

    Args:
        volumes: Image array, shape (N, ...).
        mode: "per_image" scales each volume independently;
              "cohort_global" uses global_min/max from training set.
        global_min: Training set global min (required for cohort_global).
        global_max: Training set global max (required for cohort_global).

    Returns:
        (normalized_volumes, used_min, used_max)
    """
    if mode == "per_image":
        out = np.empty_like(volumes, dtype=np.float64)
        used_min = float(volumes.min())
        used_max = float(volumes.max())
        for i in range(volumes.shape[0]):
            vmin = volumes[i].min()
            vmax = volumes[i].max()
            denom = vmax - vmin
            if denom > 0:
                out[i] = (volumes[i] - vmin) / denom
            else:
                out[i] = 0.0
        return out, used_min, used_max

    elif mode == "cohort_global":
        if global_min is None:
            global_min = float(volumes.min())
        if global_max is None:
            global_max = float(volumes.max())
        denom = global_max - global_min
        if denom > 0:
            out = np.clip((volumes - global_min) / denom, 0.0, 1.0).astype(np.float64)
        else:
            out = np.zeros_like(volumes, dtype=np.float64)
        return out, global_min, global_max

    else:
        raise ValueError(f"Unknown normalization mode: {mode!r}")


def normalize_volume(
    volume: np.ndarray,
    method: str = "zscore",
    clip_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Normalize a volume (2D or 3D).

    Args:
        volume: Input array.
        method: Normalization method — "zscore", "minmax", or "none".
        clip_range: Optional (low, high) percentile clipping before normalization.

    Returns:
        Normalized array (float64).
    """
    vol = volume.astype(np.float64)

    if clip_range is not None:
        lo, hi = np.percentile(vol[vol > 0], clip_range) if (vol > 0).any() else (0.0, 1.0)
        vol = np.clip(vol, lo, hi)

    if method == "zscore":
        mean = vol.mean()
        std = vol.std()
        if std > 0:
            vol = (vol - mean) / std
        else:
            vol = vol - mean
    elif method == "minmax":
        vmin = vol.min()
        vmax = vol.max()
        denom = vmax - vmin
        if denom > 0:
            vol = (vol - vmin) / denom
        else:
            vol = np.zeros_like(vol)
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    return vol


def resize_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, ...],
) -> np.ndarray:
    """Resize a volume to target shape using nearest-neighbor interpolation.

    Uses scipy.ndimage for interpolation. For production imaging pipelines,
    consider MONAI's transforms instead.

    Args:
        volume: Input array (2D or 3D).
        target_shape: Desired output shape.

    Returns:
        Resized array.
    """
    from scipy.ndimage import zoom

    factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
    return zoom(volume, factors, order=1)


def random_flip(
    volume: np.ndarray,
    axes: Optional[Tuple[int, ...]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Randomly flip a volume along specified axes.

    Args:
        volume: Input array.
        axes: Axes to consider for flipping. If None, all spatial axes.
        seed: Random seed for reproducibility.

    Returns:
        Possibly flipped array.
    """
    rng = np.random.default_rng(seed)
    if axes is None:
        axes = tuple(range(volume.ndim))

    out = volume.copy()
    for ax in axes:
        if rng.random() > 0.5:
            out = np.flip(out, axis=ax)
    return np.ascontiguousarray(out)


def random_noise(
    volume: np.ndarray,
    std: float = 0.01,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add Gaussian noise to a volume.

    Args:
        volume: Input array.
        std: Standard deviation of noise.
        seed: Random seed.

    Returns:
        Noisy array.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, std, volume.shape)
    return volume + noise
