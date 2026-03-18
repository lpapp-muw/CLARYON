"""Abstract base class for quantum state encodings."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class EncodingInfo:
    """Metadata about an encoding operation.

    Attributes:
        n_features_in: Number of input features before encoding.
        encoded_dim: Dimensionality of the encoded vector.
        n_qubits: Number of qubits required.
        extra: Arbitrary extra metadata from the encoding.
    """

    n_features_in: int
    encoded_dim: int
    n_qubits: int
    extra: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


class QuantumEncoding(abc.ABC):
    """Abstract base class for quantum state encoding strategies.

    Subclasses implement ``encode()`` which transforms a classical feature
    matrix into a form suitable for quantum circuit input.
    """

    @abc.abstractmethod
    def encode(
        self, X: np.ndarray, fit: bool = False
    ) -> tuple[np.ndarray, EncodingInfo]:
        """Encode a classical feature matrix for quantum circuit input.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            fit: If True, learn encoding parameters from X (e.g., padding
                length). If False, use previously fitted parameters.

        Returns:
            Tuple of (encoded_X, info) where encoded_X has shape
            (n_samples, encoded_dim) and info contains encoding metadata.
        """

    @abc.abstractmethod
    def n_qubits_for(self, n_features: int) -> int:
        """Compute the number of qubits needed for a given feature count.

        Args:
            n_features: Number of input features.

        Returns:
            Required number of qubits.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable encoding name."""
