"""Simplified quantum kernel SVM — Mottonen kernel with linear prediction.

Ported from Moradi et al. 2022 (sqKSVM.py). Computes mu = y_train @ K(X_train, X_test)
and thresholds at 0 for binary classification. Despite the name, this is closer to a
kernel GP without noise than a true SVM — it has no SVC wrapper (see HF-015).
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


@register("model", "sq_kernel_svm")
class SimplifiedQuantumKernelSVM(ModelBuilder):
    """Simplified quantum kernel SVM using Mottonen state preparation.

    Kernel: prepare |x1> via Mottonen, apply adjoint Mottonen of |x2>,
    measure projector |0><0|. Prediction: mu = y_train @ K(train, test),
    classify by sign.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        seed: int = 0,
        shots: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._n_qubits = int(n_qubits)
        self._seed = int(seed)
        self._shots = shots
        self._X_train: Optional[np.ndarray] = None
        self._y_encoded: Optional[np.ndarray] = None
        self._pl_ready = False
        self._kernel_qnode: Any = None

    @property
    def name(self) -> str:
        return "sq_kernel_svm"

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

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        # Encode labels: class 1 -> +1, class 0 -> -1
        self._y_encoded = np.where(y == 1, 1.0, -1.0)
        self._X_train = X.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None or self._y_encoded is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float64)
        K = self._kernel_matrix(self._X_train, X)
        mu = self._y_encoded @ K  # shape (n_test,)
        # Convert scores to probabilities via sigmoid
        probs_pos = 1.0 / (1.0 + np.exp(-mu))
        return np.column_stack([1.0 - probs_pos, probs_pos])

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        dump({
            "X_train": self._X_train, "y_encoded": self._y_encoded,
            "n_qubits": self._n_qubits, "seed": self._seed,
        }, model_dir / "model.joblib")

    def load(self, model_dir: Path) -> None:
        payload = joblib_load(model_dir / "model.joblib")
        self._X_train = payload["X_train"]
        self._y_encoded = payload["y_encoded"]
        self._n_qubits = payload["n_qubits"]
        self._seed = payload["seed"]
