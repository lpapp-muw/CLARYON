"""Quantum Gaussian Process classifier — Mottonen kernel with full GP inference.

Ported from Moradi et al. 2023 (GP.py). Full GP: K(train,train) + noise,
K(train,test), posterior mean+cov, sigmoid for classification. Uses Mottonen
state preparation for the quantum kernel. See HF-014.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from joblib import dump, load as joblib_load

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)


@register("model", "quantum_gp")
class QuantumGaussianProcess(ModelBuilder):
    """Quantum Gaussian Process classifier with Mottonen kernel.

    Kernel: Mottonen state prep + adjoint + projector measurement.
    GP inference: posterior mean via K(train,train)^-1 + noise regularization,
    sigmoid approximation for binary classification probabilities.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        noise: float = 0.4,
        seed: int = 0,
        shots: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._n_qubits = int(n_qubits)
        self._noise = float(noise)
        self._seed = int(seed)
        self._shots = shots
        self._X_train: Optional[np.ndarray] = None
        self._y_encoded: Optional[np.ndarray] = None
        self._K_train_inv: Optional[np.ndarray] = None
        self._pl_ready = False
        self._kernel_qnode: Any = None

    @property
    def name(self) -> str:
        return "quantum_gp"

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
        dev = qml.device("default.qubit", wires=n, shots=self._shots)
        projector = np.zeros((2**n, 2**n))
        projector[0, 0] = 1

        @qml.qnode(dev)
        def kernel_qnode(x1: np.ndarray, x2: np.ndarray) -> Any:
            qml.templates.MottonenStatePreparation(x1, wires=range(n))
            qml.adjoint(qml.templates.MottonenStatePreparation)(x2, wires=range(n))
            return qml.expval(qml.Hermitian(projector, wires=range(n)))

        self._kernel_qnode = kernel_qnode
        self._pl_ready = True

    def _kernel_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        self._init_pl()
        return np.array([[float(self._kernel_qnode(a, b)) for b in B] for a in A])

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self._X_train = X.copy()
        # Encode labels: class 1 -> +1, class 0 -> -1
        self._y_encoded = np.where(y == 1, 1.0, -1.0)
        # Precompute K(train, train) + noise and its inverse
        K_train = self._kernel_matrix(X, X)
        K_train += (self._noise ** 2) * np.eye(len(X))
        self._K_train_inv = np.linalg.inv(K_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None or self._y_encoded is None or self._K_train_inv is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float64)

        K_cross = self._kernel_matrix(self._X_train, X)  # (n_train, n_test)
        K_test = self._kernel_matrix(X, X)  # (n_test, n_test)

        # Posterior mean and covariance
        mu = K_cross.T @ self._K_train_inv @ self._y_encoded
        cov = K_test + (self._noise ** 2) * np.eye(len(X)) - K_cross.T @ self._K_train_inv @ K_cross
        var = np.diag(cov)

        # Sigmoid approximation with variance scaling (Moradi et al.)
        kappa = 1.0 / np.sqrt(1.0 + np.pi * np.sqrt(np.abs(var)) / 8.0)
        probs_pos = np.array([self._sigmoid(k * m) for k, m in zip(kappa, mu)])
        return np.column_stack([1.0 - probs_pos, probs_pos])

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        dump({
            "X_train": self._X_train, "y_encoded": self._y_encoded,
            "K_train_inv": self._K_train_inv,
            "n_qubits": self._n_qubits, "noise": self._noise, "seed": self._seed,
        }, model_dir / "model.joblib")

    def load(self, model_dir: Path) -> None:
        payload = joblib_load(model_dir / "model.joblib")
        self._X_train = payload["X_train"]
        self._y_encoded = payload["y_encoded"]
        self._K_train_inv = payload["K_train_inv"]
        self._n_qubits = payload["n_qubits"]
        self._noise = payload["noise"]
        self._seed = payload["seed"]
