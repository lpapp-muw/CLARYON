"""Conformal prediction — stub."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..registry import register
from .base import Explainer


@register("explainer", "conformal")
class ConformalExplainer(Explainer):
    """Conformal prediction stub — not yet implemented."""

    @property
    def name(self) -> str:
        return "conformal"

    def explain(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Conformal prediction not yet implemented")
