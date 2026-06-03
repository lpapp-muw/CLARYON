"""Quantum Distance Classifier — Hadamard test variant.

Ported from Moradi et al. 2022 (qDC_Hadamard_Test.py + qDS.py). Uses an ancilla qubit
with controlled Mottonen state preparation and a Hadamard test to compute similarity
between states. Class-separated: max similarity per class determines prediction.
Requires n+1 qubits (log2(features) + 1 ancilla). See HF-016, HF-017.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from joblib import dump, load as joblib_load

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)


@register("model", "qdc_hadamard")
class QuantumDistanceClassifierHadamard(ModelBuilder):
    """Quantum distance classifier using Hadamard test for similarity.

    Circuit: ancilla (wire 0) + n data qubits.
    H(ancilla) -> ctrl-Mottonen(x1) -> X(ancilla) -> ctrl-Mottonen(x2) -> X(ancilla) -> H(ancilla)
    Measure PauliZ on ancilla. Higher expectation = more similar.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        seed: int = 0,
        shots: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # n_qubits here is the data qubits (log2 of feature dim)
        # Total circuit qubits = n_qubits + 1 (ancilla)
        self._n_qubits = int(n_qubits)
        self._seed = int(seed)
        self._shots = shots
        self._class_data: dict[int, np.ndarray] = {}
        self._pl_ready = False
        self._circuit: Any = None

    @property
    def name(self) -> str:
        return "qdc_hadamard"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR_QUANTUM

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY,)

    def _init_pl(self) -> None:
        if self._pl_ready:
            return
        import pennylane as qml

        n_data = self._n_qubits
        n_total = n_data + 1  # ancilla + data
        dev = qml.device("default.qubit", wires=n_total, shots=self._shots)
        data_wires = list(range(1, n_total))

        def ops(x: np.ndarray) -> None:
            qml.templates.MottonenStatePreparation(x, wires=data_wires)

        ctrl_ops = qml.ctrl(ops, control=0)

        @qml.qnode(dev)
        def circuit(x1: np.ndarray, x2: np.ndarray) -> Any:
            qml.Hadamard(wires=0)
            ctrl_ops(x1)
            qml.PauliX(wires=0)
            ctrl_ops(x2)
            qml.PauliX(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit
        self._pl_ready = True

    def _similarity_matrix(self, class_samples: np.ndarray, test_samples: np.ndarray) -> np.ndarray:
        """Compute similarity between each class sample and each test sample."""
        self._init_pl()
        n_class, n_test = class_samples.shape[0], test_samples.shape[0]
        D = np.zeros((n_class, n_test))
        for i in range(n_class):
            for j in range(n_test):
                D[i, j] = float(self._circuit(class_samples[i], test_samples[j]))
        return D

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        classes = np.unique(y)
        self._class_data = {int(c): X[y == c].copy() for c in classes}

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._class_data:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float64)
        n_test = X.shape[0]
        classes = sorted(self._class_data.keys())
        n_classes = len(classes)

        # Compute max similarity per class for each test sample
        scores = np.zeros((n_test, n_classes))
        for ci, c in enumerate(classes):
            D = self._similarity_matrix(self._class_data[c], X)
            scores[:, ci] = np.max(D, axis=0)

        # Softmax to get probabilities
        scores_shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return probs

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        dump({
            "class_data": self._class_data,
            "n_qubits": self._n_qubits, "seed": self._seed,
        }, model_dir / "model.joblib")

    def load(self, model_dir: Path) -> None:
        payload = joblib_load(model_dir / "model.joblib")
        self._class_data = payload["class_data"]
        self._n_qubits = payload["n_qubits"]
        self._seed = payload["seed"]
