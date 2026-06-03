"""Projected quantum kernel SVM — local Pauli measurements + classical RBF.

Instead of the global fidelity kernel K[i,j] = |⟨x_i|x_j⟩|² which suffers
from exponential concentration (Thanasilp et al. 2024), this model measures
local single-qubit Pauli observables {X, Y, Z} on each qubit after amplitude
encoding, producing a classical feature vector v(x) ∈ ℝ^{3n}. A classical
RBF kernel on these Pauli vectors yields a higher-rank, more expressive kernel
matrix (Huang et al. 2021, Section 4).

Key advantages over fidelity kernel:
    - O(N) circuit evaluations instead of O(N²).
    - Kernel effective rank scales as O(n) instead of collapsing to ~2.
    - RBF bandwidth γ is tunable via presets.

Reference: Huang et al. "Power of data in quantum machine learning."
           Nature Communications 12, 2631 (2021).
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


def _compute_pauli_vectors(
    X: np.ndarray,
    n_qubits: int,
    shots: Optional[int],
) -> np.ndarray:
    """Compute Pauli expectation vectors for each sample.

    For each sample x, amplitude-encode it and measure ⟨X_k⟩, ⟨Y_k⟩, ⟨Z_k⟩
    on every qubit k, producing a feature vector in ℝ^{3·n_qubits}.

    Args:
        X: Amplitude-encoded feature matrix, shape (N, 2^n_qubits).
        n_qubits: Number of qubits.
        shots: Number of shots for measurement (None = exact).

    Returns:
        Pauli feature matrix, shape (N, 3 * n_qubits).
    """
    import pennylane as qml

    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    # Build list of observables: X_0, Y_0, Z_0, X_1, Y_1, Z_1, ...
    observables = []
    for w in wires:
        observables.append(qml.PauliX(w))
        observables.append(qml.PauliY(w))
        observables.append(qml.PauliZ(w))

    @qml.qnode(dev)
    def pauli_circuit(x: np.ndarray) -> list:
        qml.AmplitudeEmbedding(features=x, wires=wires, normalize=True)
        return [qml.expval(obs) for obs in observables]

    n_samples = X.shape[0]
    n_obs = 3 * n_qubits
    V = np.zeros((n_samples, n_obs), dtype=np.float64)

    for i in range(n_samples):
        vals = pauli_circuit(X[i])
        V[i] = np.array([float(v) for v in vals])

    return V


@register("model", "projected_kernel_svm")
class ProjectedQuantumKernelSVM(ModelBuilder):
    """Projected quantum kernel SVM with local Pauli measurements.

    Amplitude-encodes each sample, measures single-qubit Pauli observables
    {X, Y, Z} on every qubit to obtain a classical feature vector v(x) ∈ ℝ^{3n},
    then trains an RBF-kernel SVM on these Pauli vectors.

    Args:
        n_qubits: Number of qubits (set automatically by pipeline from encoding).
        gamma: RBF kernel bandwidth. If ``"auto"``, uses 1/(3n * var(V_train)).
            Accepts float or ``"auto"``.
        C: SVM regularization parameter.
        seed: Random seed for SVM.
        shots: Measurement shots (None = analytic).
    """

    def __init__(
        self,
        n_qubits: int = 4,
        gamma: Any = "auto",
        C: float = 1.0,
        seed: int = 0,
        shots: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._n_qubits = int(n_qubits)
        self._gamma_param = gamma
        self._gamma_value: Optional[float] = None
        self._C = float(C)
        self._seed = int(seed)
        self._shots = shots
        self._svc: Optional[SVC] = None
        self._V_train: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "projected_kernel_svm"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR_QUANTUM

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY,)

    def _resolve_gamma(self, V: np.ndarray) -> float:
        """Resolve RBF gamma from parameter or data.

        Args:
            V: Pauli feature matrix, shape (N, 3n).

        Returns:
            Gamma value for RBF kernel.
        """
        if self._gamma_param == "auto" or self._gamma_param is None:
            n_features = V.shape[1]
            variance = np.var(V)
            if variance < 1e-12:
                return 1.0
            return 1.0 / (n_features * variance)
        return float(self._gamma_param)

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        """Train projected quantum kernel SVM.

        Args:
            X: Amplitude-encoded feature matrix, shape (N, 2^n_qubits).
            y: Binary labels (0/1).
            task_type: Must be BINARY.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)

        logger.info(
            "  PQK: computing Pauli vectors for %d training samples "
            "(%d qubits -> %d Pauli features)...",
            X.shape[0], self._n_qubits, 3 * self._n_qubits,
        )
        V_train = _compute_pauli_vectors(X, self._n_qubits, self._shots)
        self._V_train = V_train

        self._gamma_value = self._resolve_gamma(V_train)
        logger.info("  PQK: gamma=%.6f, C=%.4f", self._gamma_value, self._C)

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
        """Predict class probabilities via projected quantum kernel.

        Args:
            X: Amplitude-encoded feature matrix.

        Returns:
            Probability matrix, shape (n_samples, 2).
        """
        if self._svc is None or self._V_train is None or self._gamma_value is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float64)

        V_test = _compute_pauli_vectors(X, self._n_qubits, self._shots)
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
            "C": self._C,
            "n_qubits": self._n_qubits,
            "seed": self._seed,
        }, model_dir / "model.joblib")

    def load(self, model_dir: Path) -> None:
        """Load model artifacts."""
        payload = joblib_load(model_dir / "model.joblib")
        self._svc = payload["svc"]
        self._V_train = payload["V_train"]
        self._gamma_value = payload["gamma_value"]
        self._gamma_param = payload["gamma_param"]
        self._C = payload["C"]
        self._n_qubits = payload["n_qubits"]
        self._seed = payload["seed"]
