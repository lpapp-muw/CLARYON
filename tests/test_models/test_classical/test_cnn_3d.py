"""Smoke test for 3D CNN — ≤3 layers, ≤5 epochs, synthetic volumes."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.io.base import TaskType


def test_cnn_3d_fit_predict(synthetic_3d_volumes):
    torch = pytest.importorskip("torch")
    from claryon.models.classical.cnn_3d import CNN3DModel

    X_tr = synthetic_3d_volumes["X_train"]
    y_tr = synthetic_3d_volumes["y_train"]
    X_te = synthetic_3d_volumes["X_test"]

    m = CNN3DModel(n_conv_layers=2, base_filters=4, epochs=3, batch_size=5, seed=42)
    m.fit(X_tr, y_tr, TaskType.BINARY)

    probs = m.predict_proba(X_te)
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    preds = m.predict(X_te)
    assert preds.shape == (5,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_cnn_3d_registered():
    pytest.importorskip("torch")
    from claryon.models.classical.cnn_3d import CNN3DModel
    from claryon.registry import get
    assert get("model", "cnn_3d") is CNN3DModel
