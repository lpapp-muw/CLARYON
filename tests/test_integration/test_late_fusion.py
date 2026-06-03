"""Integration test: late fusion — tabular MLP + 2D CNN → ensemble prediction."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.io.base import TaskType
from claryon.models.ensemble import ensemble_predictions


def test_late_fusion_tabular_plus_images(tabular_binary_Xy_train, tabular_binary_Xy_test, synthetic_2d_images):
    """Late fusion: train MLP on tabular + CNN on images, ensemble predictions."""
    pytest.importorskip("torch")
    from claryon.models.classical.mlp_ import MLPModel
    from claryon.models.classical.cnn_2d import CNN2DModel

    X_tab_tr, y_tab_tr = tabular_binary_Xy_train
    X_tab_te, y_tab_te = tabular_binary_Xy_test

    X_img_tr = synthetic_2d_images["X_train"]
    y_img_tr = synthetic_2d_images["y_train"]
    X_img_te = synthetic_2d_images["X_test"]

    # Train tabular model
    mlp = MLPModel(hidden_layer_sizes=(16,), max_iter=50, random_state=42)
    mlp.fit(X_tab_tr, y_tab_tr, TaskType.BINARY)
    tab_probs = mlp.predict_proba(X_tab_te)

    # Train image model
    cnn = CNN2DModel(n_conv_layers=2, base_filters=8, epochs=3, batch_size=10, seed=42)
    cnn.fit(X_img_tr, y_img_tr, TaskType.BINARY)
    img_probs = cnn.predict_proba(X_img_te)

    # Late fusion: average probabilities
    # Note: in a real scenario, test sets would be the same samples.
    # Here we just verify the ensemble mechanism works with differently-sized outputs.
    min_n = min(len(tab_probs), len(img_probs))
    tab_probs_aligned = tab_probs[:min_n]
    img_probs_aligned = img_probs[:min_n]

    fused_preds, fused_probs = ensemble_predictions(
        [tab_probs_aligned, img_probs_aligned], TaskType.BINARY,
    )

    assert fused_preds.shape == (min_n,)
    assert fused_probs.shape == (min_n, 2)
    assert np.allclose(fused_probs.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(fused_preds)).issubset({0, 1})
