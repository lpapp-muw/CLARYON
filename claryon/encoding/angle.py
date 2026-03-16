"""Angle encoding — map each feature to a rotation angle on a qubit.

New module. Each feature is encoded as a rotation angle, requiring one qubit per feature.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..registry import register
from .base import EncodingInfo, QuantumEncoding

logger = logging.getLogger(__name__)


@register("encoding", "angle")
class AngleEncoding(QuantumEncoding):
    """Angle encoding for quantum circuits.

    Each feature is mapped to a rotation angle on a dedicated qubit.
    Number of qubits equals number of features. Features are scaled
    to [0, pi] by default.
    """

    def __init__(self, scale_range: tuple[float, float] = (0.0, np.pi)) -> None:
        self._scale_range = scale_range
        self._fitted_n_features: Optional[int] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None

    def encode(
        self, X: np.ndarray, fit: bool = False,
    ) -> tuple[np.ndarray, EncodingInfo]:
        """Encode features as rotation angles.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            fit: If True, learn min/max from X for scaling.

        Returns:
            (encoded_X, info) where encoded_X has same shape as X,
            with values scaled to [0, pi].
        """
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            self._fitted_n_features = X.shape[1]
            self._min = X.min(axis=0)
            self._max = X.max(axis=0)

        if self._min is not None and self._max is not None:
            denom = self._max - self._min
            denom[denom == 0] = 1.0
            X_scaled = (X - self._min) / denom
            lo, hi = self._scale_range
            X_enc = X_scaled * (hi - lo) + lo
        else:
            X_enc = X

        n_qubits = X_enc.shape[1]
        info = EncodingInfo(
            n_features_in=X.shape[1],
            encoded_dim=X_enc.shape[1],
            n_qubits=n_qubits,
        )
        return X_enc, info

    def n_qubits_for(self, n_features: int) -> int:
        """One qubit per feature.

        Args:
            n_features: Number of input features.

        Returns:
            Same as n_features.
        """
        return n_features

    @property
    def name(self) -> str:
        """Encoding name."""
        return "angle"
