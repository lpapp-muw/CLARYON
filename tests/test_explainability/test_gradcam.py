"""Tests for claryon.explainability.gradcam — GradCAM stub."""
from __future__ import annotations

import numpy as np

from claryon.explainability.gradcam import GradCAMExplainer


def test_gradcam_no_crash():
    exp = GradCAMExplainer()
    X = np.random.randn(3, 1, 32, 32)
    result = exp.explain(lambda x: np.zeros(x.shape[0]), X)
    assert "heatmaps" in result
    assert result["heatmaps"].shape == X.shape


def test_gradcam_registered():
    from claryon.registry import get
    assert get("explainer", "gradcam") is GradCAMExplainer
