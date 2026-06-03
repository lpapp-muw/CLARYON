"""Tests for claryon.encoding.base — QuantumEncoding ABC."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.encoding.base import EncodingInfo, QuantumEncoding


class DummyEncoding(QuantumEncoding):
    """Minimal concrete implementation for testing the ABC."""

    def encode(self, X, fit=False):
        info = EncodingInfo(
            n_features_in=X.shape[1],
            encoded_dim=X.shape[1],
            n_qubits=X.shape[1],
        )
        return X.copy(), info

    def n_qubits_for(self, n_features):
        return n_features

    @property
    def name(self):
        return "dummy"


def test_subclass_instantiation():
    enc = DummyEncoding()
    assert enc.name == "dummy"


def test_encode_returns_info():
    enc = DummyEncoding()
    X = np.random.randn(5, 3)
    X_enc, info = enc.encode(X, fit=True)
    assert X_enc.shape == (5, 3)
    assert info.n_features_in == 3
    assert info.n_qubits == 3


def test_n_qubits_for():
    enc = DummyEncoding()
    assert enc.n_qubits_for(10) == 10


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        QuantumEncoding()  # type: ignore[abstract]


def test_encoding_info_defaults():
    info = EncodingInfo(n_features_in=5, encoded_dim=8, n_qubits=3)
    assert info.extra == {}
