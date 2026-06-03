"""Smoke test for 2D CNN — ≤3 layers, ≤5 epochs, synthetic data."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.io.base import TaskType


def test_cnn_2d_fit_predict(synthetic_2d_images):
    pytest.importorskip("torch")
    from claryon.models.classical.cnn_2d import CNN2DModel

    X_tr = synthetic_2d_images["X_train"]
    y_tr = synthetic_2d_images["y_train"]
    X_te = synthetic_2d_images["X_test"]

    m = CNN2DModel(n_conv_layers=2, base_filters=8, epochs=3, batch_size=10, seed=42)
    m.fit(X_tr, y_tr, TaskType.BINARY)

    probs = m.predict_proba(X_te)
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    preds = m.predict(X_te)
    assert preds.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_cnn_2d_registered():
    pytest.importorskip("torch")
    from claryon.models.classical.cnn_2d import CNN2DModel
    from claryon.registry import get
    assert get("model", "cnn_2d") is CNN2DModel
