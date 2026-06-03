"""ModernNCA model builder — stub.

Requires TALENT repo integration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder


@register("model", "modernnca")
class ModernNCAModel(ModelBuilder):
    """ModernNCA stub — not yet implemented."""

    @property
    def name(self) -> str:
        return "modernnca"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY, TaskType.MULTICLASS)

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        raise NotImplementedError("ModernNCA requires TALENT repo integration")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)

    def load(self, model_dir: Path) -> None:
        pass
