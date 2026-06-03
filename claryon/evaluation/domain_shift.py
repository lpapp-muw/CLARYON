"""Domain shift quantification — KS tests, center classifier, MMD, UMAP.

Provides functions for measuring distributional differences between centers
in multi-center imaging studies.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def ks_pairwise(
    features: np.ndarray,
    center_ids: np.ndarray,
    alpha: float = 0.05,
    bonferroni: bool = True,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Pairwise 2-sample KS tests across center pairs, per feature.

    For each pair of centers and each feature column, runs a two-sample
    Kolmogorov-Smirnov test. Reports the fraction of features with
    statistically significant distributional differences.

    Args:
        features: Feature matrix, shape (n_samples, n_features).
        center_ids: Center identifier per sample, shape (n_samples,).
        alpha: Significance level before correction.
        bonferroni: If True, apply Bonferroni correction (alpha / n_features).

    Returns:
        Dict mapping (center_a, center_b) to result dict with keys:
        'n_shifted', 'n_features', 'pct_shifted', 'shifted_features'.
    """
    centers = sorted(np.unique(center_ids))
    n_features = features.shape[1]
    adjusted_alpha = alpha / n_features if bonferroni else alpha

    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ca, cb in combinations(centers, 2):
        mask_a = center_ids == ca
        mask_b = center_ids == cb
        shifted = []
        for f_idx in range(n_features):
            _, p_value = stats.ks_2samp(
                features[mask_a, f_idx], features[mask_b, f_idx],
            )
            if p_value < adjusted_alpha:
                shifted.append(f_idx)

        results[(ca, cb)] = {
            "n_shifted": len(shifted),
            "n_features": n_features,
            "pct_shifted": 100.0 * len(shifted) / n_features if n_features > 0 else 0.0,
            "shifted_features": shifted,
        }

    logger.info(
        "KS pairwise: %d center pairs, bonferroni=%s, alpha=%.4f",
        len(results), bonferroni, adjusted_alpha,
    )
    return results


def center_classifier_bacc(
    X: np.ndarray,
    center_ids: np.ndarray,
    n_trees: int = 300,
    max_depth: int = 5,
    cv_folds: int = 5,
    seed: int = 42,
) -> Tuple[float, float]:
    """Train a Random Forest to predict center label from features.

    Uses stratified k-fold CV with balanced class weights to measure
    how distinguishable centers are in feature space. Lower BACC indicates
    more domain-invariant representations.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        center_ids: Center identifier per sample, shape (n_samples,).
        n_trees: Number of trees in the Random Forest.
        max_depth: Maximum tree depth.
        cv_folds: Number of CV folds.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean_bacc, std_bacc). Chance level = 1/n_centers.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(center_ids)

    clf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    baccs = []
    for train_idx, test_idx in skf.split(X, y_encoded):
        clf.fit(X[train_idx], y_encoded[train_idx])
        preds = clf.predict(X[test_idx])
        baccs.append(balanced_accuracy_score(y_encoded[test_idx], preds))

    mean_bacc = float(np.mean(baccs))
    std_bacc = float(np.std(baccs))
    logger.info(
        "Center classifier: BACC=%.3f ± %.3f (chance=%.3f)",
        mean_bacc, std_bacc, 1.0 / len(le.classes_),
    )
    return mean_bacc, std_bacc


def mmd_pairwise(
    X: np.ndarray,
    center_ids: np.ndarray,
    gamma: Optional[float] = None,
) -> Dict[Tuple[str, str], float]:
    """RBF kernel MMD² between each center pair.

    Computes the squared Maximum Mean Discrepancy with an RBF kernel
    for every unordered pair of centers.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        center_ids: Center identifier per sample, shape (n_samples,).
        gamma: RBF kernel bandwidth. Default 1/n_features.

    Returns:
        Dict mapping (center_a, center_b) to MMD² value (float).
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    centers = sorted(np.unique(center_ids))
    results: Dict[Tuple[str, str], float] = {}

    for ca, cb in combinations(centers, 2):
        Xa = X[center_ids == ca]
        Xb = X[center_ids == cb]
        mmd2 = _rbf_mmd2(Xa, Xb, gamma)
        results[(ca, cb)] = float(mmd2)
        # Store symmetric key too for convenience
        results[(cb, ca)] = float(mmd2)

    logger.info("MMD pairwise: %d unique pairs computed", len(centers) * (len(centers) - 1) // 2)
    return results


def _rbf_mmd2(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """Compute squared MMD with RBF kernel between two sample sets.

    MMD²(X,Y) = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]

    Args:
        X: First sample set, shape (n, d).
        Y: Second sample set, shape (m, d).
        gamma: RBF kernel parameter.

    Returns:
        Unbiased estimate of MMD².
    """
    from scipy.spatial.distance import cdist

    XX = cdist(X, X, "sqeuclidean")
    YY = cdist(Y, Y, "sqeuclidean")
    XY = cdist(X, Y, "sqeuclidean")

    Kxx = np.exp(-gamma * XX)
    Kyy = np.exp(-gamma * YY)
    Kxy = np.exp(-gamma * XY)

    n = Kxx.shape[0]
    m = Kyy.shape[0]

    # Unbiased estimator: exclude diagonal for same-set terms
    sum_kxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) if n > 1 else 0.0
    sum_kyy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) if m > 1 else 0.0
    sum_kxy = Kxy.mean()

    return sum_kxx - 2.0 * sum_kxy + sum_kyy


try:
    import umap  # noqa: F401

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def umap_by_center(
    X: np.ndarray,
    center_ids: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute 2D UMAP embedding with center and label metadata.

    Does NOT create a matplotlib figure — returns data for downstream
    plotting.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        center_ids: Center identifier per sample, shape (n_samples,).
        labels: Optional class labels per sample, shape (n_samples,).
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys 'embedding' (n_samples, 2), 'center_ids',
        'labels' (None if not provided).

    Raises:
        ImportError: If umap-learn is not installed.
    """
    if not HAS_UMAP:
        raise ImportError(
            "umap-learn is required for UMAP embedding. "
            "Install with: pip install umap-learn"
        )

    import umap as umap_lib

    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=seed,
    )
    embedding = reducer.fit_transform(X)

    logger.info("UMAP embedding: shape %s", embedding.shape)
    return {
        "embedding": embedding,
        "center_ids": center_ids,
        "labels": labels,
    }
