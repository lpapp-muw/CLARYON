"""QCNN MUW variant — WORK IN PROGRESS, NOT REGISTERED.

⚠️  WORK IN PROGRESS - CLARYON v0.13.0 ⚠️

This module is shipped with CLARYON but is INTENTIONALLY NOT auto-registered
with the model registry (the @register decorator below is commented out, and
the module is excluded from claryon/pipeline.py and claryon/inference.py).

This implementation is under active development and has not yet been
validated for scientific use. A revised implementation will be released in
a future CLARYON version. Until then:

  - DO NOT use this model for scientific results
  - DO NOT cite this model in publications
  - Importing this module triggers a UserWarning at import time

For working amplitude-encoded quantum models in v0.13.0, use:
    kernel_svm, projected_kernel_svm, qdc_hadamard, quantum_gp, qnn

----------------------------------------------------------------------
Original module description follows.

QCNN MUW variant — conv+pool layers with ArbitraryUnitary final block.

Ported from [E] pl_qcnn_muw.py. Input X must be amplitude-encoded.
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


# --- WIP quarantine: emit a loud warning if anyone imports this module ---
import warnings as _warnings  # noqa: E402

_warnings.warn(
    "claryon.models.quantum.qcnn_muw is WORK IN PROGRESS in CLARYON v0.13.0 "
    "and is NOT registered with the model registry. This implementation has "
    "not yet been validated for scientific use. Do not use for scientific "
    "results or publication. See README WIP notice.",
    UserWarning,
    stacklevel=2,
)
# --- end WIP quarantine ---


# @register("model", "qcnn_muw")  # WIP — see top-of-module banner. Decorator
# intentionally disabled in v0.13.0. The class definition below is preserved
# so direct instantiation continues to work for ongoing development by the
# maintainer.
class QCNNMuwModel(ModelBuilder):
    """QCNN MUW-style model with conv+pool layers and ArbitraryUnitary.

    Input X must be amplitude-encoded (L2-normalized, padded to 2^n_qubits).
    Binary classification only via projector measurement.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        seed: int = 0,
        shots: Optional[int] = None,
        n_layers: Optional[int] = None,
        init_scale: float = 0.1,
        epochs: int = 10,
        lr: float = 0.02,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> None:
        self._n_qubits = int(n_qubits)
        self._seed = int(seed)
        self._shots = shots
        self._n_layers = n_layers
        self._init_scale = float(init_scale)
        self._epochs = int(epochs)
        self._lr = float(lr)
        self._batch_size = int(batch_size)

        self._w_kernel: Any = None
        self._w_last: Any = None
        self._pl_ready = False
        self._qnode: Any = None
        self._n_final: Optional[int] = None

    @property
    def name(self) -> str:
        return "qcnn_muw"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR_QUANTUM

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY,)

    @staticmethod
    def _auto_layers(n_qubits: int, min_final: int = 2) -> int:
        n = int(n_qubits)
        layers = 0
        while n > min_final:
            n = (n + 1) // 2
            layers += 1
        return max(layers, 1)

    @staticmethod
    def _conv(qml: Any, w15: Any, wires: list[int], skip_first: bool) -> None:
        n = len(wires)
        if n < 2:
            return
        for idx in range(1 if skip_first else 0, n, 2):
            w = wires[idx]
            w2 = wires[(idx + 1) % n]
            qml.U3(*w15[0:3], wires=w)
            qml.U3(*w15[3:6], wires=w2)
            qml.IsingXX(w15[6], wires=[w, w2])
            qml.IsingYY(w15[7], wires=[w, w2])
            qml.IsingZZ(w15[8], wires=[w, w2])
            qml.U3(*w15[9:12], wires=w)
            qml.U3(*w15[12:15], wires=w2)

    @staticmethod
    def _pool(qml: Any, w3: Any, wires: list[int]) -> None:
        if len(wires) < 2:
            return
        for idx in range(1, len(wires), 2):
            ctrl = wires[idx]
            tgt = wires[idx - 1]
            qml.ctrl(qml.Rot, control=ctrl)(w3[0], w3[1], w3[2], wires=tgt)

    def _init_pl(self) -> None:
        if self._pl_ready:
            return
        import pennylane as qml

        dev = qml.device("default.qubit", wires=self._n_qubits, shots=self._shots)
        layers = self._n_layers if self._n_layers is not None else self._auto_layers(self._n_qubits)

        n_active = self._n_qubits
        for _ in range(layers):
            n_active = (n_active + 1) // 2
        self._n_final = int(n_active)

        @qml.qnode(dev, interface="autograd")
        def qnode(x: Any, wk: Any, wl: Any) -> Any:
            wires0 = list(range(self._n_qubits))
            qml.AmplitudeEmbedding(features=x, wires=wires0, normalize=True)
            active = wires0
            for layer_idx in range(layers):
                self._conv(qml, wk[layer_idx][:15], active, skip_first=(layer_idx != 0))
                self._pool(qml, wk[layer_idx][15:], active)
                active = active[::2]
            qml.ArbitraryUnitary(wl, wires=active)
            return qml.expval(qml.Projector([1], wires=[active[0]]))

        self._qnode = qnode
        self._pl_ready = True

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        """Train QCNN MUW model."""
        import pennylane as qml
        import pennylane.numpy as pnp

        rng = np.random.default_rng(self._seed)
        self._init_pl()
        layers = self._n_layers if self._n_layers is not None else self._auto_layers(self._n_qubits)

        scale = self._init_scale
        self._w_kernel = pnp.array(rng.normal(0, scale, (layers, 18)), requires_grad=True)
        last_len = (4 ** self._n_final) - 1
        self._w_last = pnp.array(rng.normal(0, scale, (last_len,)), requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=self._lr)
        N = X.shape[0]
        bs = min(self._batch_size, N)

        for epoch in range(self._epochs):
            logger.debug("qcnn_muw epoch %d/%d", epoch + 1, self._epochs)
            idx = rng.permutation(N)
            for start in range(0, N, bs):
                b = idx[start:start + bs]
                xb, yb = X[b], y[b]

                def cost(wk: Any, wl: Any) -> Any:
                    losses = pnp.stack([
                        -(yb[i] * pnp.log(self._qnode(xb[i], wk, wl) + 1e-8) +
                          (1 - yb[i]) * pnp.log(1 - self._qnode(xb[i], wk, wl) + 1e-8))
                        for i in range(len(b))
                    ])
                    return pnp.mean(losses)

                self._w_kernel, self._w_last = opt.step(cost, self._w_kernel, self._w_last)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict binary probabilities."""
        if self._w_kernel is None:
            raise RuntimeError("Model not fitted")
        self._init_pl()
        X = np.asarray(X, dtype=np.float64)
        p1 = np.array([float(self._qnode(X[i], self._w_kernel, self._w_last)) for i in range(X.shape[0])])
        return np.stack([1.0 - p1, p1], axis=1)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        np.savez(model_dir / "weights.npz",
                 w_kernel=np.array(self._w_kernel, dtype=np.float64),
                 w_last=np.array(self._w_last, dtype=np.float64))

    def load(self, model_dir: Path) -> None:
        import pennylane.numpy as pnp
        w = np.load(model_dir / "weights.npz")
        self._w_kernel = pnp.array(w["w_kernel"], requires_grad=False)
        self._w_last = pnp.array(w["w_last"], requires_grad=False)
