"""GradCAM explainer — stub for CNN models."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..registry import register
from .base import Explainer


@register("explainer", "gradcam")
class GradCAMExplainer(Explainer):
    """GradCAM stub — registered, returns placeholder data."""

    @property
    def name(self) -> str:
        return "gradcam"

    def explain(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return placeholder GradCAM output.

        Args:
            predict_fn: Model prediction function.
            X: Input data.
            feature_names: Ignored for GradCAM.

        Returns:
            Dict with placeholder 'heatmaps' of zeros.
        """
        return {
            "heatmaps": np.zeros_like(X),
            "note": "GradCAM not yet implemented — placeholder output",
        }
