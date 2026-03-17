"""Quantum Distance Classifier — SWAP test variant.

Ported from Moradi et al. 2022 (qDC_Swap_Test.py). Uses two registers for state
preparation plus an ancilla with CSWAP gates to compute state overlap. Class-separated
prediction via max similarity. Requires 2n+1 qubits (see HF-017).
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


@register("model", "qdc_swap")
class QuantumDistanceClassifierSwap(ModelBuilder):
    """Quantum distance classifier using SWAP test for similarity.

    Circuit: ancilla (wire 0) + register A (wires 1..n) + register B (wires n+1..2n).
    H(ancilla) -> Mottonen(x1, reg_A) -> Mottonen(x2, reg_B) ->
    CSWAP(ancilla, A_i, B_i) for each qubit pair -> H(ancilla) -> measure Z(ancilla).
    """

    def __init__(
        self,
        n_qubits: int = 4,
        seed: int = 0,
        shots: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # n_qubits = data qubits per register (log2 of feature dim)
        # Total circuit qubits = 2*n_qubits + 1
        self._n_qubits = int(n_qubits)
        self._seed = int(seed)
        self._shots = shots
        self._class_data: dict[int, np.ndarray] = {}
        self._pl_ready = False
        self._circuit: Any = None

    @property
    def name(self) -> str:
        return "qdc_swap"

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

        n = self._n_qubits
        n_total = 2 * n + 1
        dev = qml.device("default.qubit", wires=n_total, shots=self._shots)
        reg_a = list(range(1, n + 1))
        reg_b = list(range(n + 1, 2 * n + 1))

        @qml.qnode(dev)
        def circuit(x1: np.ndarray, x2: np.ndarray) -> Any:
            qml.Hadamard(wires=0)
            qml.templates.MottonenStatePreparation(x1, wires=reg_a)
            qml.templates.MottonenStatePreparation(x2, wires=reg_b)
            for a_i, b_i in zip(reg_a, reg_b):
                qml.CSWAP(wires=[0, a_i, b_i])
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit
        self._pl_ready = True

    def _similarity_matrix(self, class_samples: np.ndarray, test_samples: np.ndarray) -> np.ndarray:
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
