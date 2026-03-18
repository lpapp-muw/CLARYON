"""Tests for claryon.encoding.angle — angle encoding."""
from __future__ import annotations

import numpy as np

from claryon.encoding.angle import AngleEncoding


def test_angle_encoding_basic():
    enc = AngleEncoding()
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    X_enc, info = enc.encode(X, fit=True)
    assert X_enc.shape == (2, 3)
    assert info.n_qubits == 3
    assert info.n_features_in == 3
    # Values should be in [0, pi]
    assert X_enc.min() >= 0.0
    assert X_enc.max() <= np.pi + 1e-10


def test_angle_encoding_n_qubits_for():
    enc = AngleEncoding()
    assert enc.n_qubits_for(5) == 5
    assert enc.n_qubits_for(10) == 10


def test_angle_encoding_registered():
    from claryon.registry import get
    cls = get("encoding", "angle")
    assert cls is AngleEncoding


def test_angle_encoding_preserves_shape():
    enc = AngleEncoding()
    X = np.random.randn(10, 7)
    X_enc, info = enc.encode(X, fit=True)
    assert X_enc.shape == (10, 7)
