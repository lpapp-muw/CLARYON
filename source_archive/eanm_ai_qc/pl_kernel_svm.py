from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json

import numpy as np
from joblib import dump, load as joblib_load
from sklearn.svm import SVC


class PLAmplitudeKernelSVM:
    """Amplitude-encoding quantum kernel + classical SVM (PennyLane simulator).

    Notes:
    - Input X must already be amplitude-encoded vectors of length pad_len == 2**n_qubits.
    - The kernel is k(x, y) = |<x|y>|^2, evaluated by a PennyLane QNode.
    """

    name = "pl_kernel_svm"

    def __init__(
        self,
        n_qubits: int,
        pad_len: int,
        seed: int = 0,
        shots: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.pad_len = int(pad_len)
        self.seed = int(seed)
        self.shots = shots
        self.verbose = bool(verbose)

        self.svc: Optional[SVC] = None
        self.X_ref: Optional[np.ndarray] = None

        self._pl_ready = False
        self._dev = None
        self._kernel_qnode = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _init_pl(self) -> None:
        if self._pl_ready:
            return

        import pennylane as qml

        wires = list(range(self.n_qubits))
        self._dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)

        @qml.qnode(self._dev)
        def kernel_qnode(x1: np.ndarray, x2: np.ndarray) -> Any:
            # Prepare |x1>
            qml.AmplitudeEmbedding(features=x1, wires=wires, normalize=True)
            # Measure projector onto |x2>, giving |<x2|x1>|^2
            return qml.expval(qml.Projector(x2, wires=wires))

        self._kernel_qnode = kernel_qnode
        self._pl_ready = True

    def _kernel_matrix(self, A: np.ndarray, B: np.ndarray, symmetric: bool = False) -> np.ndarray:
        self._init_pl()
        assert self._kernel_qnode is not None

        nA, nB = int(A.shape[0]), int(B.shape[0])
        K = np.zeros((nA, nB), dtype=np.float64)

        kind = "train(K_train)" if symmetric else "cross(K_test)"
        self._log(f"[pl_kernel_svm] building kernel matrix {kind}: A={nA} B={nB} qubits={self.n_qubits}")

        # Print about ~10 updates at most
        step = max(1, nA // 10)

        if symmetric:
            for i in range(nA):
                K[i, i] = float(self._kernel_qnode(A[i], B[i]))
                for j in range(i + 1, nB):
                    v = float(self._kernel_qnode(A[i], B[j]))
                    K[i, j] = v
                    K[j, i] = v
                if (i + 1) % step == 0 or i == nA - 1:
                    self._log(f"[pl_kernel_svm] kernel progress: {i+1}/{nA} rows")
            return K

        for i in range(nA):
            for j in range(nB):
                K[i, j] = float(self._kernel_qnode(A[i], B[j]))
            if (i + 1) % step == 0 or i == nA - 1:
                self._log(f"[pl_kernel_svm] kernel progress: {i+1}/{nA} rows")
        return K

    def fit(self, X: np.ndarray, y01: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y01 = np.asarray(y01, dtype=int)

        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != self.pad_len:
            raise ValueError(f"X width {X.shape[1]} != pad_len {self.pad_len}")

        K = self._kernel_matrix(X, X, symmetric=True)
        self.svc = SVC(kernel="precomputed", probability=True, random_state=self.seed)
        self.svc.fit(K, y01)
        self.X_ref = X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.svc is None or self.X_ref is None:
            raise RuntimeError("Model not fitted/loaded")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != self.pad_len:
            raise ValueError(f"X width {X.shape[1]} != pad_len {self.pad_len}")

        K = self._kernel_matrix(X, self.X_ref, symmetric=False)
        return self.svc.predict_proba(K)

    def save(self, model_dir: Path, metadata: Dict[str, Any]) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        payload = {
            "svc": self.svc,
            "X_ref": self.X_ref,
            "n_qubits": self.n_qubits,
            "pad_len": self.pad_len,
            "seed": self.seed,
            "shots": self.shots,
        }
        dump(payload, model_dir / "model.joblib")

    @staticmethod
    def load(model_dir: Path) -> "PLAmplitudeKernelSVM":
        model_dir = Path(model_dir)
        _ = json.loads((model_dir / "metadata.json").read_text())
        payload = joblib_load(model_dir / "model.joblib")

        m = PLAmplitudeKernelSVM(
            n_qubits=int(payload["n_qubits"]),
            pad_len=int(payload["pad_len"]),
            seed=int(payload.get("seed", 0)),
            shots=payload.get("shots", None),
            verbose=True,
        )
        m.svc = payload["svc"]
        m.X_ref = payload["X_ref"]
        return m
