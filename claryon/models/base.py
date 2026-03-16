"""Abstract base class for all model builders."""
from __future__ import annotations

import abc
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..io.base import TaskType

logger = logging.getLogger(__name__)


class InputType(str, Enum):
    """Declares what kind of data a model expects."""

    TABULAR = "tabular"
    IMAGE_2D = "image_2d"
    IMAGE_3D = "image_3d"
    TABULAR_QUANTUM = "tabular_quantum"


class ModelBuilder(abc.ABC):
    """Abstract base class for all CLARYON models.

    Every model (classical, quantum, imaging) implements this interface.
    Models are registered via ``@register("model", "name")`` and discovered
    automatically by the pipeline.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique model identifier (matches the registry key)."""

    @property
    @abc.abstractmethod
    def input_type(self) -> InputType:
        """Type of input data this model expects."""

    @property
    @abc.abstractmethod
    def supports_tasks(self) -> tuple[TaskType, ...]:
        """Task types this model supports."""

    @abc.abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: TaskType,
        **kwargs: Any,
    ) -> None:
        """Train the model.

        Args:
            X: Feature matrix or image batch.
            y: Target array (integer labels or continuous values).
            task_type: The learning task type.
            **kwargs: Model-specific extra arguments.
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (classification) or values (regression).

        Args:
            X: Feature matrix or image batch.

        Returns:
            Predicted labels/values, shape (n_samples,).
        """

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix or image batch.

        Returns:
            Probability matrix, shape (n_samples, n_classes).

        Raises:
            NotImplementedError: For regression models.
        """

    @abc.abstractmethod
    def save(self, model_dir: Path) -> None:
        """Save model artifacts to a directory.

        Args:
            model_dir: Target directory (will be created if needed).
        """

    @abc.abstractmethod
    def load(self, model_dir: Path) -> None:
        """Load model artifacts from a directory.

        Args:
            model_dir: Directory containing saved artifacts.
        """

    def explain(self, X: np.ndarray, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Model-specific explainability (optional).

        Args:
            X: Feature matrix.
            **kwargs: Explainer-specific arguments.

        Returns:
            Dict of explanation artifacts, or None if not implemented.
        """
        return None
