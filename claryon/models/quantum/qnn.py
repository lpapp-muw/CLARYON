"""Quantum Neural Network — per-class variational circuits with margin loss.

Ported from Moradi et al. 2023 (qnn_1.py). Replaces the VQC stub. Creates one
quantum circuit per class (one-vs-rest), each with Rot+CNOT layers and trainable
weights. Uses PyTorch interface with margin-based loss. See HF-018.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)


@register("model", "qnn")
class QuantumNeuralNetwork(ModelBuilder):
    """Quantum neural network with per-class variational circuits.

    Each class has its own circuit with Mottonen state preparation followed
    by parameterized Rot+CNOT layers. Trained with margin-based loss via
    PyTorch/Adam. Prediction: argmax over class circuit scores.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 4,
        epochs: int = 200,
        lr: float = 0.001,
        batch_size: int = 100,
        margin: float = 0.15,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self._n_qubits = int(n_qubits)
        self._n_layers = int(n_layers)
        self._epochs = int(epochs)
        self._lr = float(lr)
        self._batch_size = int(batch_size)
        self._margin = float(margin)
        self._seed = int(seed)
        self._n_classes: int = 2
        self._all_weights: Optional[list] = None
        self._all_bias: Optional[list] = None
        self._qnodes: Optional[list] = None
        self._fitted = False

    @property
    def name(self) -> str:
        return "qnn"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR_QUANTUM

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY, TaskType.MULTICLASS)

    def _build_circuits(self) -> list:
        import pennylane as qml

        n = self._n_qubits
        dev = qml.device("default.qubit", wires=n)

        def layer(W: Any) -> None:
            for i in range(n):
                qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
            for j in range(n - 1):
                qml.CNOT(wires=[j, j + 1])
            if n >= 2:
                qml.CNOT(wires=[n - 1, 0])

        def circuit(weights: Any, feat: Any = None) -> Any:
            qml.templates.MottonenStatePreparation(feat, range(n))
            for W in weights:
                layer(W)
            return qml.expval(qml.PauliZ(0))

        qnodes = []
        for _ in range(self._n_classes):
            qnode = qml.QNode(circuit, dev, interface="torch")
            qnodes.append(qnode)
        return qnodes

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        import torch
        import torch.optim as optim

        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self._n_classes = int(np.max(y)) + 1

        self._qnodes = self._build_circuits()
        n = self._n_qubits

        # Initialize parameters
        all_weights = [
            torch.tensor(
                0.1 * np.random.randn(self._n_layers, n, 3),
                dtype=torch.float64, requires_grad=True,
            )
            for _ in range(self._n_classes)
        ]
        all_bias = [
            torch.tensor(
                0.1 * np.ones(1),
                dtype=torch.float64, requires_grad=True,
            )
            for _ in range(self._n_classes)
        ]

        optimizer = optim.Adam(all_weights + all_bias, lr=self._lr)
        num_train = len(y)

        for it in range(self._epochs):
            batch_idx = np.random.randint(0, num_train, (min(self._batch_size, num_train),))
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            optimizer.zero_grad()
            loss = self._compute_loss(
                self._qnodes, all_weights, all_bias, X_batch, y_batch,
            )
            loss.backward()
            optimizer.step()

            if (it + 1) % 50 == 0 or it == 0:
                logger.info("  QNN epoch %d/%d loss=%.6f", it + 1, self._epochs, loss.item())

        # Store as numpy for serialization
        self._all_weights = [w.detach().numpy().copy() for w in all_weights]
        self._all_bias = [b.detach().numpy().copy() for b in all_bias]
        self._fitted = True

    def _compute_loss(
        self,
        qnodes: list,
        all_weights: list,
        all_bias: list,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Any:
        import torch

        loss = torch.tensor(0.0, dtype=torch.float64)
        n_samples = len(y)

        for i in range(n_samples):
            feat = X[i]
            true_label = int(y[i])
            s_true = qnodes[true_label](all_weights[true_label], feat=feat) + all_bias[true_label]
            s_true = s_true.float()

            for j in range(self._n_classes):
                if j != true_label:
                    s_j = qnodes[j](all_weights[j], feat=feat) + all_bias[j]
                    s_j = s_j.float()
                    loss = loss + torch.max(
                        torch.zeros(1, dtype=torch.float32),
                        s_j - s_true + self._margin,
                    )

        return loss / n_samples

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        import torch

        X = np.asarray(X, dtype=np.float64)
        if self._qnodes is None:
            self._qnodes = self._build_circuits()

        weights_torch = [torch.tensor(w, dtype=torch.float64) for w in self._all_weights]
        bias_torch = [torch.tensor(b, dtype=torch.float64) for b in self._all_bias]

        n_test = X.shape[0]
        scores = np.zeros((n_test, self._n_classes))

        for i in range(n_test):
            for c in range(self._n_classes):
                with torch.no_grad():
                    s = float(self._qnodes[c](weights_torch[c], feat=X[i]) + bias_torch[c])
                scores[i, c] = s

        # Softmax
        scores_shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return probs

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        from joblib import dump
        dump({
            "all_weights": self._all_weights, "all_bias": self._all_bias,
            "n_qubits": self._n_qubits, "n_layers": self._n_layers,
            "n_classes": self._n_classes, "seed": self._seed,
            "margin": self._margin,
        }, model_dir / "model.joblib")

    def load(self, model_dir: Path) -> None:
        from joblib import load as joblib_load
        payload = joblib_load(model_dir / "model.joblib")
        self._all_weights = payload["all_weights"]
        self._all_bias = payload["all_bias"]
        self._n_qubits = payload["n_qubits"]
        self._n_layers = payload["n_layers"]
        self._n_classes = payload["n_classes"]
        self._seed = payload["seed"]
        self._margin = payload["margin"]
        self._fitted = True
        self._qnodes = self._build_circuits()
