"""Tests for claryon.explainability.shap_ — SHAP explainer."""
from __future__ import annotations

import pytest


def test_shap_on_classical_model(tabular_binary_Xy_train, tabular_binary_Xy_test):
    pytest.importorskip("shap")
    from claryon.explainability.shap_ import SHAPExplainer
    from claryon.models.classical.mlp_ import MLPModel
    from claryon.io.base import TaskType

    X_tr, y_tr = tabular_binary_Xy_train
    X_te, y_te = tabular_binary_Xy_test

    model = MLPModel(hidden_layer_sizes=(16,), max_iter=50, random_state=42)
    model.fit(X_tr, y_tr, TaskType.BINARY)

    explainer = SHAPExplainer(max_features=5, max_test_samples=3, background_samples=10)
    result = explainer.explain(
        predict_fn=lambda x: model.predict_proba(x)[:, 1],
        X=X_te,
        feature_names=[f"f{i}" for i in range(X_te.shape[1])],
        X_train=X_tr,
    )

    assert "shap_values" in result
    assert result["shap_values"].shape[0] == 3  # max_test_samples
    assert result["shap_values"].shape[1] == 5  # max_features
    assert len(result["feature_names"]) == 5
    assert len(result["mean_abs_shap"]) == 5


def test_shap_on_quantum_model(synthetic_amplitude_data):
    pytest.importorskip("shap")
    pytest.importorskip("pennylane")
    from claryon.explainability.shap_ import SHAPExplainer
    from claryon.models.quantum.kernel_svm import QuantumKernelSVM
    from claryon.io.base import TaskType

    X_tr = synthetic_amplitude_data["X_train"][:10]
    y_tr = synthetic_amplitude_data["y_train"][:10]
    X_te = synthetic_amplitude_data["X_test"][:3]

    model = QuantumKernelSVM(n_qubits=synthetic_amplitude_data["n_qubits"], seed=42)
    model.fit(X_tr, y_tr, TaskType.BINARY)

    explainer = SHAPExplainer(max_features=8, max_test_samples=2, background_samples=5)
    result = explainer.explain(
        predict_fn=lambda x: model.predict_proba(x)[:, 1],
        X=X_te,
        feature_names=[f"q{i}" for i in range(X_te.shape[1])],
        X_train=X_tr,
    )

    assert "shap_values" in result
    assert result["shap_values"].shape[0] == 2


def test_shap_registered():
    from claryon.explainability.shap_ import SHAPExplainer
    from claryon.registry import get
    assert get("explainer", "shap") is SHAPExplainer
