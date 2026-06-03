"""Explainability utilities — feature variance selection for reduced-space explanation.

Ported from [E] utils.py.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def select_feature_indices_by_variance(
    X: np.ndarray,
    max_features: Optional[int],
) -> np.ndarray:
    """Select top-variance feature indices.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        max_features: Maximum number of features to keep. If None or >= n_features,
            returns all indices.

    Returns:
        Sorted array of selected feature indices.
    """
    n_features = X.shape[1]
    if max_features is None or max_features >= n_features:
        return np.arange(n_features, dtype=int)
    v = np.var(X, axis=0)
    idx = np.argsort(v)[::-1][:max_features]
    return np.sort(idx)


def subset_features(
    X: np.ndarray,
    feature_names: List[str],
    idx: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Subset features by index.

    Args:
        X: Feature matrix.
        feature_names: Column names.
        idx: Feature indices to keep.

    Returns:
        Tuple of (X_subset, names_subset).
    """
    Xs = X[:, idx]
    names = [feature_names[i] for i in idx.tolist()]
    return Xs, names
