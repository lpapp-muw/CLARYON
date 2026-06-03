"""CatBoost model builder."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)


@register("model", "catboost")
class CatBoostModel(ModelBuilder):
    """CatBoost gradient boosting model."""

    def __init__(self, **params: Any) -> None:
        self._params = {
            "iterations": 1000,
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": 0,
        }
        self._params.update(params)
        self._model: Any = None
        self._task_type: TaskType = TaskType.BINARY

    @property
    def name(self) -> str:
        return "catboost"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION)

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        """Train CatBoost model."""
        from catboost import CatBoostClassifier, CatBoostRegressor

        self._task_type = task_type
        params = dict(self._params)

        if task_type == TaskType.REGRESSION:
            params.pop("auto_class_weights", None)
            self._model = CatBoostRegressor(**params)
        else:
            self._model = CatBoostClassifier(**params)

        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values."""
        preds = self._model.predict(X)
        if self._task_type != TaskType.REGRESSION:
            preds = np.asarray(preds).flatten().astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self._task_type == TaskType.REGRESSION:
            raise NotImplementedError("predict_proba not available for regression")
        return self._model.predict_proba(X)

    def save(self, model_dir: Path) -> None:
        """Save model."""
        model_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(model_dir / "model.cbm"))

    def load(self, model_dir: Path) -> None:
        """Load model."""
        from catboost import CatBoostClassifier, CatBoostRegressor

        if self._task_type == TaskType.REGRESSION:
            self._model = CatBoostRegressor()
        else:
            self._model = CatBoostClassifier()
        self._model.load_model(str(model_dir / "model.cbm"))
