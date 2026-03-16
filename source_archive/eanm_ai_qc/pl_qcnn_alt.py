from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json

import numpy as np


class PLQCNN_Alt:
    """QCNN (ALT-style) operating on amplitude-encoded vectors.

    - Input X must be amplitude-encoded vectors of length pad_len (=2**n_qubits).
    - Circuit: repeated (conv + pooling) layers, then StronglyEntanglingLayers on final active wires.
    - Output: P(class=1) via projector measurement on the first active wire.
    """

    name = "pl_qcnn_alt"

    def __init__(
        self,
        n_qubits: int,
        pad_len: int,
        seed: int = 0,
        shots: Optional[int] = None,
        n_layers: Optional[int] = None,
        dense_layers: int = 1,
        init_scale: float = 0.1,
        verbose: bool = True,
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.pad_len = int(pad_len)
        self.seed = int(seed)
        self.shots = shots
        self.n_layers = n_layers
        self.dense_layers = int(dense_layers)
        self.init_scale = float(init_scale)
        self.verbose = bool(verbose)

        self.w_kernel = None
        self.w_dense = None

        self._pl_ready = False
        self._dev = None
        self._qnode = None
        self._n_final = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    @staticmethod
    def _auto_layers(n_qubits: int, min_final_wires: int = 2) -> int:
        n = int(n_qubits)
        layers = 0
        while n > int(min_final_wires):
            n = (n + 1) // 2
            layers += 1
        return max(layers, 1)

    @staticmethod
    def _conv(qml, w15: Any, wires: list[int], skip_first: bool) -> None:
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
    def _pool(qml, w3: Any, wires: list[int]) -> None:
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
        import pennylane.numpy as pnp

        self._dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
        layers = self.n_layers if self.n_layers is not None else self._auto_layers(self.n_qubits, 2)

        n_active = self.n_qubits
        for _ in range(layers):
            n_active = (n_active + 1) // 2
        self._n_final = int(n_active)

        @qml.qnode(self._dev, interface="autograd")
        def qnode(x: Any, wk: Any, wd: Any) -> Any:
            wires0 = list(range(self.n_qubits))
            qml.AmplitudeEmbedding(features=x, wires=wires0, normalize=True)

            active = wires0
            for layer in range(layers):
                self._conv(qml, wk[layer][:15], active, skip_first=(layer != 0))
                self._pool(qml, wk[layer][15:], active)
                active = active[::2]

            qml.templates.StronglyEntanglingLayers(wd, wires=active)
            return qml.expval(qml.Projector([1], wires=[active[0]]))

        self._qnode = qnode
        self._pl_ready = True

    def fit(
        self,
        X: np.ndarray,
        y01: np.ndarray,
        epochs: int = 10,
        lr: float = 0.02,
        batch_size: Optional[int] = 16,
    ) -> None:
        import pennylane as qml
        import pennylane.numpy as pnp

        rng = np.random.default_rng(self.seed)

        self._init_pl()
        layers = self.n_layers if self.n_layers is not None else self._auto_layers(self.n_qubits, 2)
        assert self._n_final is not None
        if self._qnode is None:
            raise RuntimeError("QNode not initialized")

        # Small init helps avoid immediate saturation / barren-plateau-like behavior.
        scale = float(self.init_scale)
        self.w_kernel = pnp.array(rng.normal(0, scale, size=(layers, 18)), requires_grad=True)
        self.w_dense = pnp.array(
            rng.normal(0, scale, size=(int(self.dense_layers), int(self._n_final), 3)),
            requires_grad=True,
        )

        def bce(p: Any, y: Any) -> Any:
            eps = 1e-8
            return -(y * pnp.log(p + eps) + (1 - y) * pnp.log(1 - p + eps))

        opt = qml.AdamOptimizer(stepsize=float(lr))
        N = int(X.shape[0])
        if batch_size is None or int(batch_size) <= 0 or int(batch_size) > N:
            batch_size = N

        for epoch in range(int(epochs)):
            self._log(f"[pl_qcnn_alt] epoch {epoch+1}/{int(epochs)}")
            idx = rng.permutation(N)
            for start in range(0, N, int(batch_size)):
                b = idx[start : start + int(batch_size)]
                xb = X[b]
                yb = y01[b]

                def cost(wk: Any, wd: Any) -> Any:
                    losses = []
                    for i in range(xb.shape[0]):
                        p = self._qnode(xb[i], wk, wd)
                        losses.append(bce(p, yb[i]))
                    return pnp.mean(pnp.stack(losses))

                self.w_kernel, self.w_dense = opt.step(cost, self.w_kernel, self.w_dense)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w_kernel is None or self.w_dense is None:
            raise RuntimeError("Model not fitted/loaded")
        self._init_pl()
        if self._qnode is None:
            raise RuntimeError("QNode not initialized")

        X = np.asarray(X, dtype=np.float64)
        p1 = np.array(
            [float(self._qnode(X[i], self.w_kernel, self.w_dense)) for i in range(X.shape[0])],
            dtype=np.float64,
        )
        return np.stack([1.0 - p1, p1], axis=1)

    def save(self, model_dir: Path, metadata: Dict[str, Any]) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        np.savez(
            model_dir / "weights.npz",
            w_kernel=np.array(self.w_kernel, dtype=np.float64),
            w_dense=np.array(self.w_dense, dtype=np.float64),
        )

    @staticmethod
    def load(model_dir: Path) -> "PLQCNN_Alt":
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "metadata.json").read_text())
        w = np.load(model_dir / "weights.npz")
        import pennylane.numpy as pnp

        hp = meta.get("hyperparams", {}) or {}
        qcnn_layers = hp.get("qcnn_layers", None)
        n_layers = int(qcnn_layers) if qcnn_layers is not None else None

        dense_layers = hp.get("qcnn_dense_layers", None)
        dense_layers = int(dense_layers) if dense_layers is not None else int(meta.get("dense_layers", 1))

        init_scale = hp.get("qcnn_init_scale", None)
        init_scale = float(init_scale) if init_scale is not None else 0.1

        m = PLQCNN_Alt(
            n_qubits=int(meta["n_qubits"]),
            pad_len=int(meta["pad_len"]),
            seed=int(meta.get("seed", 0)),
            shots=meta.get("shots", None),
            n_layers=n_layers,
            dense_layers=dense_layers,
            init_scale=init_scale,
            verbose=True,
        )
        m.w_kernel = pnp.array(w["w_kernel"], requires_grad=False)
        m.w_dense = pnp.array(w["w_dense"], requires_grad=False)
        return m
