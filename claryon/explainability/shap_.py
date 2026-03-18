"""SHAP explainer — model-agnostic permutation SHAP with reduced-space support.

Ported from [E] shap_explain.py. Generalized beyond binary classification.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..registry import register
from .base import Explainer
from .utils import select_feature_indices_by_variance, subset_features

logger = logging.getLogger(__name__)


@register("explainer", "shap")
class SHAPExplainer(Explainer):
    """Model-agnostic SHAP with permutation explainer.

    Operates in a reduced feature space (top-variance features) for
    feasibility. The model is always called on the full feature vector
    via baseline expansion.
    """

    def __init__(
        self,
        max_features: Optional[int] = 32,
        max_test_samples: int = 5,
        background_samples: int = 20,
        max_evals: Optional[int] = None,
    ) -> None:
        self._max_features = max_features
        self._max_test_samples = max_test_samples
        self._background_samples = background_samples
        self._max_evals = max_evals

    @property
    def name(self) -> str:
        return "shap"

    def explain(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
        X_train: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate SHAP explanations.

        Args:
            predict_fn: Callable that takes (N, F) and returns (N,) or (N, C).
            X: Test samples to explain, shape (N, F).
            feature_names: Feature names.
            X_train: Training data for background. If None, uses X.
            **kwargs: Extra arguments.

        Returns:
            Dict with 'shap_values', 'feature_names', 'mean_abs_shap'.
        """
        import shap

        X = np.asarray(X, dtype=np.float64)
        if X_train is None:
            X_train = X

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

        # Reduce feature space
        idx = select_feature_indices_by_variance(X_train, self._max_features)
        Xtr_red, names_red = subset_features(X_train, feature_names, idx)
        Xte_red, _ = subset_features(X, feature_names, idx)

        baseline_full = np.mean(X_train, axis=0)

        def f_red(x_red: np.ndarray) -> np.ndarray:
            x_red = np.asarray(x_red, dtype=np.float64)
            if x_red.ndim == 1:
                x_red = x_red[None, :]
            Xfull = np.tile(baseline_full, (x_red.shape[0], 1))
            Xfull[:, idx] = x_red
            return predict_fn(Xfull)

        n_bg = min(self._background_samples, Xtr_red.shape[0])
        n_te = min(self._max_test_samples, Xte_red.shape[0])
        bg = Xtr_red[:n_bg]
        ex = Xte_red[:n_te]

        explainer = shap.PermutationExplainer(f_red, bg)

        max_evals = self._max_evals
        if max_evals is None:
            max_evals = int(min(2048, 10 * max(1, ex.shape[1])))

        try:
            shap_out = explainer(ex, max_evals=max_evals)
        except TypeError:
            shap_out = explainer(ex)

        shap_vals = shap_out.values
        mean_abs = np.mean(np.abs(shap_vals), axis=0)

        return {
            "shap_values": shap_vals,
            "feature_names": names_red,
            "feature_indices": idx,
            "mean_abs_shap": mean_abs,
        }
