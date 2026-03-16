from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class AmplitudeEncodingInfo:
    n_features_in: int
    pad_len: int
    n_qubits: int
    zero_norm_rows: int

def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << int(math.ceil(math.log2(n)))

def amplitude_encode_matrix(
    X: np.ndarray,
    pad_len: Optional[int] = None,
) -> Tuple[np.ndarray, AmplitudeEncodingInfo]:
    """Pad each row to 2^n and L2-normalize for amplitude embedding.

    This is not statistical preprocessing (no scaling/PCA). It is the minimal
    encoding required to represent a vector as quantum amplitudes.
    """
    X = np.asarray(X, dtype=np.float64)

    # sanitize NaN/Inf (quantum simulators cannot accept NaN)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, n_features = X.shape
    if pad_len is None:
        pad_len = _next_pow2(n_features)
    if pad_len < n_features:
        raise ValueError(f"pad_len {pad_len} < n_features {n_features}")

    out = np.zeros((n_samples, pad_len), dtype=np.float64)
    out[:, :n_features] = X

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    zero_mask = norms.squeeze() == 0.0
    zero_norm_rows = int(zero_mask.sum())

    norms = np.maximum(norms, 1e-12)
    out = out / norms

    # If a row is all-zero, force a valid basis state |0...0>
    if zero_norm_rows > 0:
        out[zero_mask, :] = 0.0
        out[zero_mask, 0] = 1.0

    n_qubits = int(round(math.log2(pad_len)))
    if (1 << n_qubits) != pad_len:
        raise ValueError("pad_len must be power of two")

    info = AmplitudeEncodingInfo(
        n_features_in=int(n_features),
        pad_len=int(pad_len),
        n_qubits=int(n_qubits),
        zero_norm_rows=zero_norm_rows,
    )
    return out, info
