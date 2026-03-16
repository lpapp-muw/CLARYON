"""Smoke tests for classical model builders — XGBoost, LightGBM, CatBoost, MLP."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.io.base import TaskType
from claryon.registry import clear, get


@pytest.fixture(autouse=True)
def _import_models():
    """Import classical model modules to trigger registration."""
    # Importing triggers @register decorators
    import claryon.models.classical.mlp_  # noqa: F401

    try:
        import claryon.models.classical.xgboost_  # noqa: F401
    except ImportError:
        pass
    try:
        import claryon.models.classical.lightgbm_  # noqa: F401
    except ImportError:
        pass
    try:
        import claryon.models.classical.catboost_  # noqa: F401
    except ImportError:
        pass


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((80, 10))
    y_train = np.array([0, 1] * 40)
    X_test = rng.standard_normal((20, 10))
    y_test = np.array([0, 1] * 10)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((120, 10))
    y_train = np.array([0, 1, 2] * 40)
    X_test = rng.standard_normal((30, 10))
    y_test = np.array([0, 1, 2] * 10)
    return X_train, y_train, X_test, y_test


class TestMLPModel:
    def test_fit_predict_binary(self, binary_data):
        from claryon.models.classical.mlp_ import MLPModel
        m = MLPModel(hidden_layer_sizes=(32,), max_iter=50, random_state=42)
        X_tr, y_tr, X_te, y_te = binary_data
        m.fit(X_tr, y_tr, TaskType.BINARY)
        preds = m.predict(X_te)
        assert preds.shape == (20,)
        assert set(np.unique(preds)).issubset({0, 1})
        probs = m.predict_proba(X_te)
        assert probs.shape == (20, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_save_load(self, binary_data, tmp_path):
        from claryon.models.classical.mlp_ import MLPModel
        m = MLPModel(hidden_layer_sizes=(16,), max_iter=30, random_state=42)
        X_tr, y_tr, X_te, _ = binary_data
        m.fit(X_tr, y_tr, TaskType.BINARY)
        probs1 = m.predict_proba(X_te)
        m.save(tmp_path / "mlp")
        m2 = MLPModel()
        m2.load(tmp_path / "mlp")
        probs2 = m2.predict_proba(X_te)
        np.testing.assert_array_almost_equal(probs1, probs2)

    def test_registered(self):
        cls = get("model", "mlp")
        from claryon.models.classical.mlp_ import MLPModel
        assert cls is MLPModel


class TestXGBoostModel:
    def test_fit_predict_binary(self, binary_data):
        xgb = pytest.importorskip("xgboost")
        from claryon.models.classical.xgboost_ import XGBoostModel
        m = XGBoostModel(n_estimators=10, random_state=42)
        X_tr, y_tr, X_te, _ = binary_data
        m.fit(X_tr, y_tr, TaskType.BINARY)
        preds = m.predict(X_te)
        assert preds.shape == (20,)
        probs = m.predict_proba(X_te)
        assert probs.shape == (20, 2)

    def test_multiclass(self, multiclass_data):
        xgb = pytest.importorskip("xgboost")
        from claryon.models.classical.xgboost_ import XGBoostModel
        m = XGBoostModel(n_estimators=10, random_state=42)
        X_tr, y_tr, X_te, _ = multiclass_data
        m.fit(X_tr, y_tr, TaskType.MULTICLASS)
        probs = m.predict_proba(X_te)
        assert probs.shape == (30, 3)


class TestLightGBMModel:
    def test_fit_predict_binary(self, binary_data):
        lgb = pytest.importorskip("lightgbm")
        from claryon.models.classical.lightgbm_ import LightGBMModel
        m = LightGBMModel(n_estimators=10, random_state=42)
        X_tr, y_tr, X_te, _ = binary_data
        m.fit(X_tr, y_tr, TaskType.BINARY)
        preds = m.predict(X_te)
        assert preds.shape == (20,)
        probs = m.predict_proba(X_te)
        assert probs.shape == (20, 2)


class TestCatBoostModel:
    def test_fit_predict_binary(self, binary_data):
        cb = pytest.importorskip("catboost")
        from claryon.models.classical.catboost_ import CatBoostModel
        m = CatBoostModel(iterations=10, random_seed=42, verbose=0)
        X_tr, y_tr, X_te, _ = binary_data
        m.fit(X_tr, y_tr, TaskType.BINARY)
        preds = m.predict(X_te)
        assert preds.shape == (20,)
        probs = m.predict_proba(X_te)
        assert probs.shape == (20, 2)
