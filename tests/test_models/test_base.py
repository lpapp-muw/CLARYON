"""Tests for claryon.models.base — ModelBuilder ABC."""
from __future__ import annotations


import numpy as np
import pytest

from claryon.io.base import TaskType
from claryon.models.base import InputType, ModelBuilder


class DummyModel(ModelBuilder):
    """Minimal concrete implementation for testing the ABC."""

    @property
    def name(self):
        return "dummy"

    @property
    def input_type(self):
        return InputType.TABULAR

    @property
    def supports_tasks(self):
        return (TaskType.BINARY, TaskType.MULTICLASS)

    def fit(self, X, y, task_type, **kwargs):
        self._fitted = True

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X):
        n = X.shape[0]
        return np.full((n, 2), 0.5)

    def save(self, model_dir):
        model_dir.mkdir(parents=True, exist_ok=True)

    def load(self, model_dir):
        pass


def test_subclass_instantiation():
    m = DummyModel()
    assert m.name == "dummy"
    assert m.input_type == InputType.TABULAR
    assert TaskType.BINARY in m.supports_tasks


def test_fit_predict():
    m = DummyModel()
    X = np.random.randn(10, 5)
    y = np.array([0, 1] * 5)
    m.fit(X, y, TaskType.BINARY)
    preds = m.predict(X)
    assert preds.shape == (10,)
    probs = m.predict_proba(X)
    assert probs.shape == (10, 2)


def test_save_load(tmp_path):
    m = DummyModel()
    m.save(tmp_path / "model")
    assert (tmp_path / "model").is_dir()
    m.load(tmp_path / "model")


def test_explain_default_returns_none():
    m = DummyModel()
    assert m.explain(np.zeros((1, 5))) is None


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        ModelBuilder()  # type: ignore[abstract]
