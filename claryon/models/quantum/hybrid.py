"""Hybrid quantum-classical model — stub."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...io.base import TaskType
from ...registry import register
from ..base import InputType, ModelBuilder


@register("model", "hybrid")
class HybridModel(ModelBuilder):
    """Hybrid quantum-classical stub — not yet implemented."""

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR_QUANTUM

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY,)

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: TaskType, **kwargs: Any) -> None:
        raise NotImplementedError("Hybrid model not yet implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)

    def load(self, model_dir: Path) -> None:
        pass
