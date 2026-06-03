"""Tests for claryon.explainability.lime_ — LIME explainer."""
from __future__ import annotations

import pytest


def test_lime_on_classical_model(tabular_binary_Xy_train, tabular_binary_Xy_test):
    pytest.importorskip("lime")
    from claryon.explainability.lime_ import LIMEExplainer
    from claryon.models.classical.mlp_ import MLPModel
    from claryon.io.base import TaskType

    X_tr, y_tr = tabular_binary_Xy_train
    X_te, y_te = tabular_binary_Xy_test

    model = MLPModel(hidden_layer_sizes=(16,), max_iter=50, random_state=42)
    model.fit(X_tr, y_tr, TaskType.BINARY)

    explainer = LIMEExplainer(
        max_features=5, max_test_samples=2,
        num_features_explained=5, num_samples=100,
    )
    result = explainer.explain(
        predict_fn=model.predict_proba,
        X=X_te,
        feature_names=[f"f{i}" for i in range(X_te.shape[1])],
        X_train=X_tr,
    )

    assert "explanations" in result
    assert len(result["explanations"]) == 2
    assert len(result["feature_names"]) == 5
    # Each explanation should be a dict of feature → weight
    for exp in result["explanations"]:
        assert isinstance(exp, dict)


def test_lime_registered():
    from claryon.explainability.lime_ import LIMEExplainer
    from claryon.registry import get
    assert get("explainer", "lime") is LIMEExplainer
