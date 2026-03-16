"""Abstract base class for explainability methods."""
from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Explainer(abc.ABC):
    """Abstract base class for all CLARYON explainers.

    Explainers are registered via ``@register("explainer", "name")`` and can
    be applied to any model that supports the required interface.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable explainer name."""

    @abc.abstractmethod
    def explain(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate explanations for model predictions.

        Args:
            predict_fn: A callable that takes X and returns predictions or
                probabilities. Shape depends on the explainer.
            X: Feature matrix to explain, shape (n_samples, n_features).
            feature_names: Optional feature names for labeling.
            **kwargs: Explainer-specific arguments.

        Returns:
            Dict of explanation artifacts (values, figures, etc.).
        """

    def save(self, artifacts: Dict[str, Any], output_dir: Path) -> None:
        """Save explanation artifacts to disk.

        Args:
            artifacts: Dict returned by ``explain()``.
            output_dir: Target directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, val in artifacts.items():
            if isinstance(val, np.ndarray):
                np.save(output_dir / f"{key}.npy", val)
                logger.debug("Saved %s to %s", key, output_dir / f"{key}.npy")
