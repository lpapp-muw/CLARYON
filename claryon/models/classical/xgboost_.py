"""XGBoost model builder."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)


@register("model", "xgboost")
class XGBoostModel(ModelBuilder):
    """XGBoost gradient boosting model.

    Supports binary classification, multiclass classification, and regression.
    """

    def __init__(self, **params: Any) -> None:
        self._params = {
            "n_estimators": 1000,
            "verbosity": 0,
            "random_state": 42,
        }
        self._params.update(params)
        self._model: Any = None
        self._task_type: TaskType = TaskType.BINARY

    @property
    def name(self) -> str:
        return "xgboost"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION)

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        """Train XGBoost model."""
        import xgboost as xgb

        self._task_type = task_type
        params = dict(self._params)

        if task_type == TaskType.REGRESSION:
            self._model = xgb.XGBRegressor(**params)
        else:
            params.setdefault("eval_metric", "mlogloss")
            self._model = xgb.XGBClassifier(**params)

        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values."""
        preds = self._model.predict(X)
        if self._task_type != TaskType.REGRESSION:
            preds = preds.astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self._task_type == TaskType.REGRESSION:
            raise NotImplementedError("predict_proba not available for regression")
        return self._model.predict_proba(X)

    def save(self, model_dir: Path) -> None:
        """Save model."""
        model_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(model_dir / "model.json"))

    def load(self, model_dir: Path) -> None:
        """Load model."""
        import xgboost as xgb

        if self._task_type == TaskType.REGRESSION:
            self._model = xgb.XGBRegressor()
        else:
            self._model = xgb.XGBClassifier()
        self._model.load_model(str(model_dir / "model.json"))
