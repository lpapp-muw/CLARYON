"""Tests for claryon.encoding.amplitude — amplitude encoding."""
from __future__ import annotations


import numpy as np
import pytest

from claryon.encoding.amplitude import (
    AmplitudeEncoding,
    _next_pow2,
    amplitude_encode_matrix,
)


def test_next_pow2():
    assert _next_pow2(1) == 1
    assert _next_pow2(3) == 4
    assert _next_pow2(4) == 4
    assert _next_pow2(5) == 8
    assert _next_pow2(306) == 512


def test_amplitude_encode_basic():
    X = np.array([[1.0, 2.0, 3.0]])
    X_enc, info = amplitude_encode_matrix(X)
    assert X_enc.shape == (1, 4)  # next pow2 of 3 is 4
    assert info.n_features_in == 3
    assert info.encoded_dim == 4
    assert info.n_qubits == 2
    # Verify L2 norm is 1
    np.testing.assert_almost_equal(np.linalg.norm(X_enc[0]), 1.0)


def test_amplitude_encode_preserves_direction():
    X = np.array([[3.0, 4.0]])
    X_enc, _ = amplitude_encode_matrix(X, pad_len=2)
    # Original direction: [3, 4] → [0.6, 0.8]
    np.testing.assert_almost_equal(X_enc[0, 0], 0.6)
    np.testing.assert_almost_equal(X_enc[0, 1], 0.8)


def test_amplitude_encode_zero_row():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    X_enc, info = amplitude_encode_matrix(X, pad_len=2)
    # Zero row should become |0⟩ = [1, 0]
    np.testing.assert_almost_equal(X_enc[0, 0], 1.0)
    np.testing.assert_almost_equal(X_enc[0, 1], 0.0)
    assert info.extra["zero_norm_rows"] == 1


def test_amplitude_encode_pad_len_too_small():
    X = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="pad_len 2 < n_features 3"):
        amplitude_encode_matrix(X, pad_len=2)


def test_amplitude_encode_not_power_of_two():
    X = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="power of two"):
        amplitude_encode_matrix(X, pad_len=3)


def test_amplitude_encode_nan_sanitized():
    X = np.array([[1.0, np.nan, np.inf]])
    X_enc, _ = amplitude_encode_matrix(X)
    assert not np.any(np.isnan(X_enc))
    assert not np.any(np.isinf(X_enc))


def test_amplitude_encoding_class_fit_encode():
    enc = AmplitudeEncoding()
    X = np.random.randn(10, 7)
    X_enc, info = enc.encode(X, fit=True)
    assert X_enc.shape == (10, 8)  # next pow2 of 7
    assert info.n_qubits == 3

    # Encode without fit should use fitted pad_len
    X2 = np.random.randn(5, 7)
    X2_enc, info2 = enc.encode(X2, fit=False)
    assert X2_enc.shape == (5, 8)


def test_amplitude_encoding_n_qubits_for():
    enc = AmplitudeEncoding()
    assert enc.n_qubits_for(1) == 0
    assert enc.n_qubits_for(3) == 2
    assert enc.n_qubits_for(306) == 9


def test_amplitude_encoding_registered():
    from claryon.registry import get
    cls = get("encoding", "amplitude")
    assert cls is AmplitudeEncoding
