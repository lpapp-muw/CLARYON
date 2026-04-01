"""LightGBM model builder."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)


@register("model", "lightgbm")
class LightGBMModel(ModelBuilder):
    """LightGBM gradient boosting model."""

    def __init__(self, **params: Any) -> None:
        self._params = {
            "n_estimators": 1000,
            "random_state": 42,
            "verbose": -1,
        }
        self._params.update(params)
        self._model: Any = None
        self._task_type: TaskType = TaskType.BINARY

    @property
    def name(self) -> str:
        return "lightgbm"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION)

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        """Train LightGBM model."""
        import lightgbm as lgb

        self._task_type = task_type
        params = dict(self._params)

        if task_type == TaskType.REGRESSION:
            self._model = lgb.LGBMRegressor(**params)
        else:
            self._model = lgb.LGBMClassifier(**params)

        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            preds = self._model.predict(X)
        if self._task_type != TaskType.REGRESSION:
            preds = preds.astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self._task_type == TaskType.REGRESSION:
            raise NotImplementedError("predict_proba not available for regression")
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            return self._model.predict_proba(X)

    def save(self, model_dir: Path) -> None:
        """Save model."""
        model_dir.mkdir(parents=True, exist_ok=True)
        self._model.booster_.save_model(str(model_dir / "model.txt"))

    def load(self, model_dir: Path) -> None:
        """Load model."""
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        if self._task_type == TaskType.REGRESSION:
            self._model = lgb.LGBMRegressor()
        else:
            self._model = lgb.LGBMClassifier()
        self._model._Booster = booster
        self._model.fitted_ = True
