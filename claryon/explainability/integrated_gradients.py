"""Integrated Gradients explainer — stub."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..registry import register
from .base import Explainer


@register("explainer", "integrated_gradients")
class IntegratedGradientsExplainer(Explainer):
    """Integrated Gradients stub — not yet implemented."""

    @property
    def name(self) -> str:
        return "integrated_gradients"

    def explain(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Integrated Gradients not yet implemented")
