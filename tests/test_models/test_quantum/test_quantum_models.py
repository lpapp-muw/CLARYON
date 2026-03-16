"""Smoke tests for quantum model builders — ≤4 qubits, ≤2 epochs, ≤20 samples."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def amplitude_data(synthetic_amplitude_data):
    """Extract small amplitude-encoded data for quantum smoke tests."""
    return synthetic_amplitude_data


class TestQuantumKernelSVM:
    def test_fit_predict(self, amplitude_data):
        qml = pytest.importorskip("pennylane")
        from claryon.models.quantum.kernel_svm import QuantumKernelSVM

        X_tr = amplitude_data["X_train"][:10]  # ≤20 samples
        y_tr = amplitude_data["y_train"][:10]
        X_te = amplitude_data["X_test"][:5]

        from claryon.io.base import TaskType
        m = QuantumKernelSVM(n_qubits=amplitude_data["n_qubits"], seed=42)
        m.fit(X_tr, y_tr, TaskType.BINARY)
        probs = m.predict_proba(X_te)
        assert probs.shape == (5, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=0.1)

        preds = m.predict(X_te)
        assert preds.shape == (5,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_registered(self):
        pytest.importorskip("pennylane")
        from claryon.models.quantum.kernel_svm import QuantumKernelSVM
        from claryon.registry import get
        assert get("model", "kernel_svm") is QuantumKernelSVM


class TestQCNNMuw:
    def test_fit_predict(self, amplitude_data):
        qml = pytest.importorskip("pennylane")
        from claryon.models.quantum.qcnn_muw import QCNNMuwModel
        from claryon.io.base import TaskType

        X_tr = amplitude_data["X_train"][:10]
        y_tr = amplitude_data["y_train"][:10]
        X_te = amplitude_data["X_test"][:5]

        m = QCNNMuwModel(
            n_qubits=amplitude_data["n_qubits"], seed=42,
            epochs=2, lr=0.02, batch_size=5,
        )
        m.fit(X_tr, y_tr, TaskType.BINARY)
        probs = m.predict_proba(X_te)
        assert probs.shape == (5, 2)
        preds = m.predict(X_te)
        assert preds.shape == (5,)


class TestQCNNAlt:
    def test_fit_predict(self, amplitude_data):
        qml = pytest.importorskip("pennylane")
        from claryon.models.quantum.qcnn_alt import QCNNAltModel
        from claryon.io.base import TaskType

        X_tr = amplitude_data["X_train"][:10]
        y_tr = amplitude_data["y_train"][:10]
        X_te = amplitude_data["X_test"][:5]

        m = QCNNAltModel(
            n_qubits=amplitude_data["n_qubits"], seed=42,
            epochs=2, lr=0.02, batch_size=5,
        )
        m.fit(X_tr, y_tr, TaskType.BINARY)
        probs = m.predict_proba(X_te)
        assert probs.shape == (5, 2)
        preds = m.predict(X_te)
        assert preds.shape == (5,)
