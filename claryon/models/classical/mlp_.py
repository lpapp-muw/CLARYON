"""scikit-learn MLP model builder."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)


@register("model", "mlp")
class MLPModel(ModelBuilder):
    """scikit-learn Multi-Layer Perceptron."""

    def __init__(self, **params: Any) -> None:
        self._params = {
            "hidden_layer_sizes": (128, 64),
            "max_iter": 1000,
            "random_state": 42,
        }
        self._params.update(params)
        self._model: Any = None
        self._task_type: TaskType = TaskType.BINARY

    @property
    def name(self) -> str:
        return "mlp"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION)

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        """Train MLP model."""
        self._task_type = task_type
        params = dict(self._params)

        if task_type == TaskType.REGRESSION:
            self._model = MLPRegressor(**params)
        else:
            self._model = MLPClassifier(**params)

        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values."""
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self._task_type == TaskType.REGRESSION:
            raise NotImplementedError("predict_proba not available for regression")
        return self._model.predict_proba(X)

    def save(self, model_dir: Path) -> None:
        """Save model."""
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, model_dir / "model.pkl")

    def load(self, model_dir: Path) -> None:
        """Load model."""
        self._model = joblib.load(model_dir / "model.pkl")
