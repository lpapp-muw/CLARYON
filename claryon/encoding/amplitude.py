"""Amplitude encoding — pad to 2^n and L2-normalize for quantum circuits.

Ported from [E] encoding.py. Registered as ("encoding", "amplitude").
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from ..registry import register
from .base import EncodingInfo, QuantumEncoding

logger = logging.getLogger(__name__)

MAX_QUBITS_WARNING = 20


def _next_pow2(n: int) -> int:
    """Return the smallest power of 2 >= n.

    Args:
        n: Input integer.

    Returns:
        Ceiling power of 2.
    """
    if n <= 1:
        return 1
    return 1 << int(math.ceil(math.log2(n)))


def amplitude_encode_matrix(
    X: np.ndarray,
    pad_len: Optional[int] = None,
) -> tuple[np.ndarray, EncodingInfo]:
    """Pad each row to 2^n and L2-normalize for amplitude embedding.

    This is not statistical preprocessing (no scaling/PCA). It is the minimal
    encoding required to represent a classical vector as quantum amplitudes.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        pad_len: Target dimension (must be power of 2). If None, uses the
            smallest power of 2 >= n_features.

    Returns:
        Tuple of (encoded_X, info) where encoded_X has shape
        (n_samples, pad_len) with unit L2 norm per row.

    Raises:
        ValueError: If pad_len < n_features or pad_len is not a power of 2.
    """
    X = np.asarray(X, dtype=np.float64)

    # Sanitize NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, n_features = X.shape
    if pad_len is None:
        pad_len = _next_pow2(n_features)
    if pad_len < n_features:
        raise ValueError(f"pad_len {pad_len} < n_features {n_features}")

    n_qubits = int(round(math.log2(pad_len)))
    if (1 << n_qubits) != pad_len:
        raise ValueError(f"pad_len must be power of two, got {pad_len}")

    if n_qubits > MAX_QUBITS_WARNING:
        logger.warning(
            "Amplitude encoding requires %d qubits — simulation cost O(2^%d). "
            "Consider reducing features first.",
            n_qubits, n_qubits,
        )

    # Pad
    out = np.zeros((n_samples, pad_len), dtype=np.float64)
    out[:, :n_features] = X

    # L2 normalize
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    zero_mask = norms.squeeze() == 0.0
    zero_norm_rows = int(zero_mask.sum())

    norms = np.maximum(norms, 1e-12)
    out = out / norms

    # Force zero-norm rows to |0...0⟩ basis state
    if zero_norm_rows > 0:
        out[zero_mask, :] = 0.0
        out[zero_mask, 0] = 1.0

    info = EncodingInfo(
        n_features_in=int(n_features),
        encoded_dim=int(pad_len),
        n_qubits=int(n_qubits),
        extra={"zero_norm_rows": zero_norm_rows, "pad_len": int(pad_len)},
    )
    return out, info


@register("encoding", "amplitude")
class AmplitudeEncoding(QuantumEncoding):
    """Amplitude encoding strategy for quantum circuits.

    Pads feature vectors to the next power of 2, then L2-normalizes each row
    so the vector can be used as quantum state amplitudes.
    """

    def __init__(self, pad_len: Optional[int] = None) -> None:
        self._pad_len = pad_len
        self._fitted_pad_len: Optional[int] = None

    def encode(
        self, X: np.ndarray, fit: bool = False
    ) -> tuple[np.ndarray, EncodingInfo]:
        """Encode features via amplitude embedding.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            fit: If True, determine pad_len from X. If False, reuse fitted value.

        Returns:
            (encoded_X, info) tuple.
        """
        pad = self._pad_len
        if fit:
            if pad is None:
                pad = _next_pow2(X.shape[1])
            self._fitted_pad_len = pad
        else:
            pad = self._fitted_pad_len or self._pad_len

        return amplitude_encode_matrix(X, pad_len=pad)

    def n_qubits_for(self, n_features: int) -> int:
        """Compute qubits needed for a given feature count.

        Args:
            n_features: Number of input features.

        Returns:
            Number of qubits (log2 of padded dimension).
        """
        pad = _next_pow2(n_features)
        return int(round(math.log2(pad)))

    @property
    def name(self) -> str:
        """Encoding name."""
        return "amplitude"
