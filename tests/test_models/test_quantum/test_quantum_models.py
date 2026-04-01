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
        pytest.importorskip("pennylane")
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
        pytest.importorskip("pennylane")
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
        pytest.importorskip("pennylane")
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


class TestProjectedQuantumKernelSVM:
    def test_fit_predict(self, amplitude_data):
        pytest.importorskip("pennylane")
        from claryon.models.quantum.projected_kernel_svm import ProjectedQuantumKernelSVM
        from claryon.io.base import TaskType

        X_tr = amplitude_data["X_train"][:10]
        y_tr = amplitude_data["y_train"][:10]
        X_te = amplitude_data["X_test"][:5]

        m = ProjectedQuantumKernelSVM(
            n_qubits=amplitude_data["n_qubits"], seed=42, gamma="auto",
        )
        m.fit(X_tr, y_tr, TaskType.BINARY)
        probs = m.predict_proba(X_te)
        assert probs.shape == (5, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=0.1)

        preds = m.predict(X_te)
        assert preds.shape == (5,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_save_load(self, amplitude_data, tmp_path):
        pytest.importorskip("pennylane")
        from claryon.models.quantum.projected_kernel_svm import ProjectedQuantumKernelSVM
        from claryon.io.base import TaskType

        X_tr = amplitude_data["X_train"][:10]
        y_tr = amplitude_data["y_train"][:10]
        X_te = amplitude_data["X_test"][:5]

        m = ProjectedQuantumKernelSVM(
            n_qubits=amplitude_data["n_qubits"], seed=42,
        )
        m.fit(X_tr, y_tr, TaskType.BINARY)
        probs_before = m.predict_proba(X_te)

        save_dir = tmp_path / "pqk_model"
        m.save(save_dir)

        m2 = ProjectedQuantumKernelSVM(n_qubits=amplitude_data["n_qubits"])
        m2.load(save_dir)
        probs_after = m2.predict_proba(X_te)
        assert np.allclose(probs_before, probs_after, atol=1e-6)

    def test_registered(self):
        pytest.importorskip("pennylane")
        from claryon.models.quantum.projected_kernel_svm import ProjectedQuantumKernelSVM
        from claryon.registry import get
        assert get("model", "projected_kernel_svm") is ProjectedQuantumKernelSVM


@pytest.fixture
def raw_tabular_data():
    """Small raw tabular dataset for angle-encoding models (no amplitude encoding)."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((20, 8))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = rng.standard_normal((10, 8))
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    return {"X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test}


class TestAngleProjectedKernelSVM:
    def test_fit_predict(self, raw_tabular_data):
        pytest.importorskip("pennylane")
        from claryon.models.quantum.angle_pqk_svm import AngleProjectedKernelSVM
        from claryon.io.base import TaskType

        d = raw_tabular_data
        m = AngleProjectedKernelSVM(bandwidth=1.0, seed=42)
        m.fit(d["X_train"], d["y_train"], TaskType.BINARY)
        probs = m.predict_proba(d["X_test"])
        assert probs.shape == (10, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=0.1)

        preds = m.predict(d["X_test"])
        assert preds.shape == (10,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_save_load(self, raw_tabular_data, tmp_path):
        pytest.importorskip("pennylane")
        from claryon.models.quantum.angle_pqk_svm import AngleProjectedKernelSVM
        from claryon.io.base import TaskType

        d = raw_tabular_data
        m = AngleProjectedKernelSVM(bandwidth=1.0, seed=42)
        m.fit(d["X_train"], d["y_train"], TaskType.BINARY)
        probs_before = m.predict_proba(d["X_test"])

        save_dir = tmp_path / "angle_pqk_model"
        m.save(save_dir)

        m2 = AngleProjectedKernelSVM()
        m2.load(save_dir)
        probs_after = m2.predict_proba(d["X_test"])
        assert np.allclose(probs_before, probs_after, atol=1e-6)

    def test_registered(self):
        pytest.importorskip("pennylane")
        from claryon.models.quantum.angle_pqk_svm import AngleProjectedKernelSVM
        from claryon.registry import get
        assert get("model", "angle_pqk_svm") is AngleProjectedKernelSVM
