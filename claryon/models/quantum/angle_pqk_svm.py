"""Angle-encoded projected quantum kernel SVM.

Combines angle encoding (1 qubit per feature, no L2 normalization) with
projected quantum kernel (local Pauli measurements + classical RBF).

Amplitude encoding's L2 normalization destroys magnitude information and
forces all data onto the unit hypersphere, causing Pauli expectations to
concentrate (Thanasilp et al. 2024, arXiv:2503.01545). Angle encoding
avoids this entirely: each feature independently rotates a dedicated qubit
via RY(bandwidth * x_i), preserving per-feature distance information.

The resulting kernel is structurally richer: K(x,x') is built from Pauli
vectors in R^{3n} where n = n_features (not log2(n_features)). A tunable
bandwidth parameter c controls the kernel's sensitivity — the single most
impactful hyperparameter for quantum kernels (Shaydulin & Wild, 2021).

This model uses InputType.TABULAR (not TABULAR_QUANTUM) because:
    1. Z-score standardization IS beneficial for angle encoding (unlike
       amplitude encoding where it destroys kernel geometry per HF-031).
    2. No amplitude encoding step is needed — angle encoding happens
       inside the model's QNode.
    3. mRMR + max_features preprocessing applies normally.

References:
    Huang et al. "Power of data in quantum machine learning."
        Nature Communications 12, 2631 (2021).
    Shaydulin & Wild. "Importance of kernel bandwidth in quantum
        machine learning." Physical Review A 106, 042407 (2022).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from joblib import dump, load as joblib_load
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)


def _compute_angle_pauli_vectors(
    X: np.ndarray,
    bandwidth: float,
    shots: Optional[int],
) -> np.ndarray:
    """Compute Pauli expectation vectors with angle encoding.

    For each sample x, encode features via RY(bandwidth * x_i) on qubit i,
    then measure {X_k, Y_k, Z_k} on every qubit, producing a feature
    vector in R^{3 * n_features}.

    Args:
        X: Feature matrix, shape (N, n_features). Raw (z-scored) features.
        bandwidth: Scaling factor applied to features before rotation.
        shots: Number of shots (None = analytic / exact).

    Returns:
        Pauli feature matrix, shape (N, 3 * n_features).
    """
    import pennylane as qml

    n_features = X.shape[1]
    n_qubits = n_features
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    observables = []
    for w in wires:
        observables.append(qml.PauliX(w))
        observables.append(qml.PauliY(w))
        observables.append(qml.PauliZ(w))

    @qml.qnode(dev)
    def pauli_circuit(x: np.ndarray) -> list:
        qml.AngleEmbedding(x, wires=wires, rotation="Y")
        return [qml.expval(obs) for obs in observables]

    n_samples = X.shape[0]
    n_obs = 3 * n_qubits
    V = np.zeros((n_samples, n_obs), dtype=np.float64)

    X_scaled = X * bandwidth

    for i in range(n_samples):
        vals = pauli_circuit(X_scaled[i])
        V[i] = np.array([float(v) for v in vals])

    return V


@register("model", "angle_pqk_svm")
class AngleProjectedKernelSVM(ModelBuilder):
    """Angle-encoded projected quantum kernel SVM.

    Encodes each feature as an RY rotation on a dedicated qubit (1 qubit
    per feature), measures single-qubit Pauli observables {X, Y, Z},
    then trains an RBF-kernel SVM on the resulting Pauli vectors.

    This model uses InputType.TABULAR — it receives z-scored features
    directly from the pipeline (no amplitude encoding). The quantum
    encoding happens internally.

    Args:
        bandwidth: Feature scaling before rotation. Controls kernel
            sensitivity. Higher values spread data across more of the
            Bloch sphere. Default 1.0.
        gamma: RBF kernel bandwidth. ``"auto"`` uses 1/(d * var(V)).
        C: SVM regularization parameter.
        seed: Random seed for SVM.
        shots: Measurement shots (None = analytic).
    """

    def __init__(
        self,
        bandwidth: float = 1.0,
        gamma: Any = "auto",
        C: float = 1.0,
        seed: int = 0,
        shots: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._bandwidth = float(bandwidth)
        self._gamma_param = gamma
        self._gamma_value: Optional[float] = None
        self._C = float(C)
        self._seed = int(seed)
        self._shots = shots
        self._svc: Optional[SVC] = None
        self._V_train: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    @property
    def name(self) -> str:
        return "angle_pqk_svm"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY,)

    def _resolve_gamma(self, V: np.ndarray) -> float:
        """Resolve RBF gamma from parameter or data.

        Args:
            V: Pauli feature matrix, shape (N, 3 * n_features).

        Returns:
            Gamma value for RBF kernel.
        """
        if self._gamma_param == "auto" or self._gamma_param is None:
            n_dims = V.shape[1]
            variance = np.var(V)
            if variance < 1e-12:
                return 1.0
            return 1.0 / (n_dims * variance)
        return float(self._gamma_param)

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        """Train angle-encoded projected quantum kernel SVM.

        Args:
            X: Feature matrix, shape (N, n_features). Z-scored tabular features.
            y: Binary labels (0/1).
            task_type: Must be BINARY.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self._n_features = X.shape[1]

        logger.info(
            "  Angle PQK: %d features -> %d qubits, bandwidth=%.3f, "
            "computing Pauli vectors for %d samples...",
            X.shape[1], X.shape[1], self._bandwidth, X.shape[0],
        )
        V_train = _compute_angle_pauli_vectors(X, self._bandwidth, self._shots)
        self._V_train = V_train

        self._gamma_value = self._resolve_gamma(V_train)
        logger.info(
            "  Angle PQK: Pauli dim=%d, gamma=%.6f, C=%.4f",
            V_train.shape[1], self._gamma_value, self._C,
        )

        K_train = rbf_kernel(V_train, V_train, gamma=self._gamma_value)

        self._svc = SVC(
            kernel="precomputed",
            C=self._C,
            probability=True,
            random_state=self._seed,
            class_weight="balanced",
        )
        self._svc.fit(K_train, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix (raw z-scored features).

        Returns:
            Probability matrix, shape (n_samples, 2).
        """
        if self._svc is None or self._V_train is None or self._gamma_value is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float64)

        V_test = _compute_angle_pauli_vectors(X, self._bandwidth, self._shots)
        K_test = rbf_kernel(V_test, self._V_train, gamma=self._gamma_value)
        return self._svc.predict_proba(K_test)

    def save(self, model_dir: Path) -> None:
        """Save model artifacts."""
        model_dir.mkdir(parents=True, exist_ok=True)
        dump({
            "svc": self._svc,
            "V_train": self._V_train,
            "gamma_value": self._gamma_value,
            "gamma_param": self._gamma_param,
            "bandwidth": self._bandwidth,
            "C": self._C,
            "n_features": self._n_features,
            "seed": self._seed,
        }, model_dir / "model.joblib")

    def load(self, model_dir: Path) -> None:
        """Load model artifacts."""
        payload = joblib_load(model_dir / "model.joblib")
        self._svc = payload["svc"]
        self._V_train = payload["V_train"]
        self._gamma_value = payload["gamma_value"]
        self._gamma_param = payload["gamma_param"]
        self._bandwidth = payload["bandwidth"]
        self._C = payload["C"]
        self._n_features = payload["n_features"]
        self._seed = payload["seed"]
