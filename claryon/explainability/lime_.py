"""LIME explainer — model-agnostic local explanations with reduced-space support.

Ported from [E] lime_explain.py. Generalized beyond binary classification.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..registry import register
from .base import Explainer
from .utils import select_feature_indices_by_variance, subset_features

logger = logging.getLogger(__name__)


@register("explainer", "lime")
class LIMEExplainer(Explainer):
    """Model-agnostic LIME explainer.

    Operates in a reduced feature space for feasibility.
    """

    def __init__(
        self,
        max_features: Optional[int] = 32,
        max_test_samples: int = 5,
        num_features_explained: int = 10,
        num_samples: int = 1000,
    ) -> None:
        self._max_features = max_features
        self._max_test_samples = max_test_samples
        self._num_features_explained = num_features_explained
        self._num_samples = num_samples

    @property
    def name(self) -> str:
        return "lime"

    def explain(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
        X_train: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate LIME explanations.

        Args:
            predict_fn: Callable that takes (N, F) and returns (N, C) probabilities.
            X: Test samples to explain.
            feature_names: Feature names.
            X_train: Training data. If None, uses X.
            class_names: Class label names.
            **kwargs: Extra arguments.

        Returns:
            Dict with 'explanations' list of per-sample feature weights.
        """
        from lime.lime_tabular import LimeTabularExplainer

        X = np.asarray(X, dtype=np.float64)
        if X_train is None:
            X_train = X

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

        if class_names is None:
            class_names = ["class_0", "class_1"]

        idx = select_feature_indices_by_variance(X_train, self._max_features)
        Xtr_red, names_red = subset_features(X_train, feature_names, idx)
        Xte_red, _ = subset_features(X, feature_names, idx)

        baseline_full = np.mean(X_train, axis=0)

        def predict_proba_red(x_red: np.ndarray) -> np.ndarray:
            x_red = np.asarray(x_red, dtype=np.float64)
            if x_red.ndim == 1:
                x_red = x_red[None, :]
            Xfull = np.tile(baseline_full, (x_red.shape[0], 1))
            Xfull[:, idx] = x_red
            return predict_fn(Xfull)

        explainer = LimeTabularExplainer(
            training_data=Xtr_red,
            feature_names=names_red,
            class_names=class_names,
            mode="classification",
            discretize_continuous=False,
        )

        n = min(self._max_test_samples, Xte_red.shape[0])
        explanations = []
        for i in range(n):
            exp = explainer.explain_instance(
                data_row=Xte_red[i],
                predict_fn=predict_proba_red,
                num_features=min(self._num_features_explained, Xtr_red.shape[1]),
                num_samples=self._num_samples,
            )
            sample_exp = {feat: float(w) for feat, w in exp.as_list()}
            explanations.append(sample_exp)

        return {
            "explanations": explanations,
            "feature_names": names_red,
            "feature_indices": idx,
        }
