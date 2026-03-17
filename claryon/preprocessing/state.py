"""PreprocessingState — stores all preprocessing parameters fitted on training data.

Saved per model/seed/fold. Loaded before inference to apply identical transforms.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingState:
    """Stores all preprocessing parameters fitted on training data.

    Saved per model/seed/fold. Loaded before inference.

    Attributes:
        z_mean: Per-feature mean from training data, shape (n_features_original,).
        z_std: Per-feature std from training data, shape (n_features_original,).
        selected_features: Indices into original feature space after mRMR.
        selected_feature_names: Original column names of selected features.
        spearman_threshold: Threshold used for mRMR redundancy clustering.
        image_norm_mode: "per_image" or "cohort_global".
        image_norm_min: Cohort global min from training set (if applicable).
        image_norm_max: Cohort global max from training set (if applicable).
        n_features_original: Number of features before selection.
        n_features_selected: Number of features after selection.
    """

    # Z-score normalization
    z_mean: np.ndarray
    z_std: np.ndarray

    # mRMR feature selection
    selected_features: List[int]
    selected_feature_names: List[str]
    spearman_threshold: float

    # Image normalization (for CNN/qCNN)
    image_norm_mode: str = "per_image"
    image_norm_min: Optional[float] = None
    image_norm_max: Optional[float] = None

    # Metadata
    n_features_original: int = 0
    n_features_selected: int = 0
    preprocessing_applied: str = "zscore_mrmr"  # "zscore_mrmr" or "mrmr_only"

    def save(self, path: Path) -> None:
        """Serialize preprocessing state to JSON.

        Args:
            path: Output file path (typically preprocessing_state.json).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "z_mean": self.z_mean.tolist(),
            "z_std": self.z_std.tolist(),
            "selected_features": self.selected_features,
            "selected_feature_names": self.selected_feature_names,
            "spearman_threshold": self.spearman_threshold,
            "image_norm_mode": self.image_norm_mode,
            "image_norm_min": self.image_norm_min,
            "image_norm_max": self.image_norm_max,
            "n_features_original": self.n_features_original,
            "n_features_selected": self.n_features_selected,
            "preprocessing_applied": self.preprocessing_applied,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved PreprocessingState to %s", path)

    @staticmethod
    def load(path: Path) -> PreprocessingState:
        """Deserialize preprocessing state from JSON.

        Args:
            path: Input file path.

        Returns:
            Loaded PreprocessingState instance.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return PreprocessingState(
            z_mean=np.array(data["z_mean"], dtype=np.float64),
            z_std=np.array(data["z_std"], dtype=np.float64),
            selected_features=data["selected_features"],
            selected_feature_names=data["selected_feature_names"],
            spearman_threshold=data["spearman_threshold"],
            image_norm_mode=data.get("image_norm_mode", "per_image"),
            image_norm_min=data.get("image_norm_min"),
            image_norm_max=data.get("image_norm_max"),
            n_features_original=data.get("n_features_original", 0),
            n_features_selected=data.get("n_features_selected", 0),
            preprocessing_applied=data.get("preprocessing_applied", "zscore_mrmr"),
        )

    def apply_tabular(self, X: np.ndarray) -> np.ndarray:
        """Normalize and select features using stored train coefficients.

        Applies z-score only if preprocessing_applied == "zscore_mrmr".
        Quantum models (mrmr_only) get feature selection without z-score.

        Args:
            X: Feature matrix, shape (n_samples, n_features_original).

        Returns:
            Transformed matrix, shape (n_samples, n_features_selected).
        """
        if self.preprocessing_applied == "mrmr_only":
            return X[:, self.selected_features]
        X_z = (X - self.z_mean) / np.maximum(self.z_std, 1e-12)
        return X_z[:, self.selected_features]

    def apply_image(self, volumes: np.ndarray) -> np.ndarray:
        """Normalize image volumes using stored parameters.

        Args:
            volumes: Image array, shape (N, C, ...).

        Returns:
            Normalized image array scaled to [0, 1].
        """
        if self.image_norm_mode == "per_image":
            out = np.empty_like(volumes, dtype=np.float64)
            for i in range(volumes.shape[0]):
                vmin = volumes[i].min()
                vmax = volumes[i].max()
                denom = vmax - vmin
                if denom > 0:
                    out[i] = (volumes[i] - vmin) / denom
                else:
                    out[i] = 0.0
            return out
        elif self.image_norm_mode == "cohort_global":
            gmin = self.image_norm_min if self.image_norm_min is not None else 0.0
            gmax = self.image_norm_max if self.image_norm_max is not None else 1.0
            denom = gmax - gmin
            if denom > 0:
                return np.clip((volumes - gmin) / denom, 0.0, 1.0).astype(np.float64)
            return np.zeros_like(volumes, dtype=np.float64)
        else:
            raise ValueError(f"Unknown image_norm_mode: {self.image_norm_mode!r}")
