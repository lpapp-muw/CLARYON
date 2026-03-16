"""TabPFN model builder — zero-shot tabular foundation model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)


@register("model", "tabpfn")
class TabPFNModel(ModelBuilder):
    """TabPFN zero-shot tabular foundation model.

    Requires ``pip install tabpfn``. Only supports classification.
    """

    def __init__(self, **params: Any) -> None:
        self._params = params
        self._model: Any = None
        self._task_type: TaskType = TaskType.BINARY

    @property
    def name(self) -> str:
        return "tabpfn"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY, TaskType.MULTICLASS)

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        """Fit TabPFN (mostly stores training data for in-context learning)."""
        from tabpfn import TabPFNClassifier

        self._task_type = task_type
        self._model = TabPFNClassifier(**self._params)
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self._model.predict_proba(X)

    def save(self, model_dir: Path) -> None:
        """Save model (TabPFN doesn't persist traditional weights)."""
        model_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(self._model, model_dir / "tabpfn.pkl")

    def load(self, model_dir: Path) -> None:
        """Load model."""
        import joblib
        self._model = joblib.load(model_dir / "tabpfn.pkl")
