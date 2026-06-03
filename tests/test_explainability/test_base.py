"""Tests for claryon.explainability.base — Explainer ABC."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.explainability.base import Explainer


class DummyExplainer(Explainer):
    """Minimal concrete implementation for testing the ABC."""

    @property
    def name(self):
        return "dummy_explainer"

    def explain(self, predict_fn, X, feature_names=None, **kwargs):
        preds = predict_fn(X)
        return {"attributions": np.zeros_like(X), "predictions": preds}


def test_subclass_instantiation():
    exp = DummyExplainer()
    assert exp.name == "dummy_explainer"


def test_explain_returns_dict():
    exp = DummyExplainer()
    X = np.random.randn(5, 3)
    result = exp.explain(lambda x: np.zeros(x.shape[0]), X)
    assert "attributions" in result
    assert result["attributions"].shape == (5, 3)


def test_save_artifacts(tmp_path):
    exp = DummyExplainer()
    artifacts = {"values": np.array([1.0, 2.0, 3.0])}
    exp.save(artifacts, tmp_path / "explain_out")
    assert (tmp_path / "explain_out" / "values.npy").exists()
    loaded = np.load(tmp_path / "explain_out" / "values.npy")
    np.testing.assert_array_equal(loaded, artifacts["values"])


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        Explainer()  # type: ignore[abstract]
