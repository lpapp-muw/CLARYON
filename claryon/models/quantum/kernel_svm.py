"""Quantum kernel SVM — amplitude-encoding quantum kernel + classical SVM.

Ported from [E] pl_kernel_svm.py. Input X must be amplitude-encoded.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from joblib import dump, load as joblib_load
from sklearn.svm import SVC

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)


@register("model", "kernel_svm")
class QuantumKernelSVM(ModelBuilder):
    """Amplitude-encoding quantum kernel + classical SVM.

    The kernel is k(x,y) = |<x|y>|^2, evaluated by a PennyLane QNode.
    Input X must already be amplitude-encoded (L2-normalized, padded to 2^n).
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
        self._svc: Optional[SVC] = None
        self._X_ref: Optional[np.ndarray] = None
        self._pl_ready = False
        self._kernel_qnode: Any = None

    @property
    def name(self) -> str:
        return "kernel_svm"

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

        wires = list(range(self._n_qubits))
        dev = qml.device("default.qubit", wires=self._n_qubits, shots=self._shots)

        @qml.qnode(dev)
        def kernel_qnode(x1: np.ndarray, x2: np.ndarray) -> Any:
            qml.AmplitudeEmbedding(features=x1, wires=wires, normalize=True)
            return qml.expval(qml.Projector(x2, wires=wires))

        self._kernel_qnode = kernel_qnode
        self._pl_ready = True

    def _kernel_matrix(self, A: np.ndarray, B: np.ndarray, symmetric: bool = False) -> np.ndarray:
        self._init_pl()
        nA, nB = A.shape[0], B.shape[0]
        K = np.zeros((nA, nB), dtype=np.float64)

        if symmetric:
            for i in range(nA):
                K[i, i] = float(self._kernel_qnode(A[i], B[i]))
                for j in range(i + 1, nB):
                    v = float(self._kernel_qnode(A[i], B[j]))
                    K[i, j] = v
                    K[j, i] = v
        else:
            for i in range(nA):
                for j in range(nB):
                    K[i, j] = float(self._kernel_qnode(A[i], B[j]))
        return K

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        """Train quantum kernel SVM.

        Args:
            X: Amplitude-encoded feature matrix, shape (N, 2^n_qubits).
            y: Binary labels (0/1).
            task_type: Must be BINARY.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)

        K = self._kernel_matrix(X, X, symmetric=True)
        self._svc = SVC(kernel="precomputed", probability=True, random_state=self._seed)
        self._svc.fit(K, y)
        self._X_ref = X.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via quantum kernel."""
        if self._svc is None or self._X_ref is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float64)
        K = self._kernel_matrix(X, self._X_ref, symmetric=False)
        return self._svc.predict_proba(K)

    def save(self, model_dir: Path) -> None:
        """Save model."""
        model_dir.mkdir(parents=True, exist_ok=True)
        dump({
            "svc": self._svc, "X_ref": self._X_ref,
            "n_qubits": self._n_qubits, "seed": self._seed,
        }, model_dir / "model.joblib")

    def load(self, model_dir: Path) -> None:
        """Load model."""
        payload = joblib_load(model_dir / "model.joblib")
        self._svc = payload["svc"]
        self._X_ref = payload["X_ref"]
        self._n_qubits = payload["n_qubits"]
        self._seed = payload["seed"]
