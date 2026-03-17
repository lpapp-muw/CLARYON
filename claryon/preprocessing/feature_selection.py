"""mRMR feature selection — correlation-based Minimum Redundancy Maximum Relevance.

Algorithm (Papp et al. radiomics variant, NOT Peng mutual-information mRMR):
1. Compute Spearman rank correlation matrix on training features
2. Cluster features by redundancy: |ρ| > threshold → same cluster
3. Within each cluster, keep the feature with highest |Spearman ρ| to the target
4. Return selected feature indices
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def mrmr_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    spearman_threshold: float = 0.8,
    max_features: Optional[int] = None,
) -> Tuple[List[int], List[str]]:
    """Minimum Redundancy Maximum Relevance feature selection.

    Args:
        X_train: Training features (n_samples, n_features).
        y_train: Training labels (n_samples,).
        feature_names: Column names for logging.
        spearman_threshold: Features with |ρ| > threshold are redundant. Default 0.8.
        max_features: Optional hard cap on output features.

    Returns:
        (selected_indices, selected_names)
    """
    n_features = X_train.shape[1]

    # Guard: if n_features <= 4, skip mRMR
    if n_features <= 4:
        logger.info("mRMR skipped: n_features=%d <= 4", n_features)
        indices = list(range(n_features))
        names = feature_names[:n_features]
        return indices, names

    # Step 1: Compute feature-to-target relevance (|Spearman ρ|)
    relevance = np.zeros(n_features)
    for j in range(n_features):
        col = X_train[:, j]
        if np.std(col) < 1e-12:
            relevance[j] = 0.0
        else:
            rho, _ = spearmanr(col, y_train)
            relevance[j] = abs(rho)

    # Step 2: Compute feature-feature Spearman correlation matrix
    corr_matrix = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if np.std(X_train[:, i]) < 1e-12 or np.std(X_train[:, j]) < 1e-12:
                corr_matrix[i, j] = 0.0
            else:
                rho, _ = spearmanr(X_train[:, i], X_train[:, j])
                corr_matrix[i, j] = abs(rho)
            corr_matrix[j, i] = corr_matrix[i, j]

    # Step 3: Build redundancy clusters using union-find
    parent = list(range(n_features))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n_features):
        for j in range(i + 1, n_features):
            if corr_matrix[i, j] > spearman_threshold:
                union(i, j)

    # Step 4: Group features by cluster
    from collections import defaultdict
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n_features):
        clusters[find(i)].append(i)

    # Step 5: Within each cluster, keep the feature with highest relevance
    selected_indices: List[int] = []
    for cluster_members in clusters.values():
        best_idx = max(cluster_members, key=lambda i: relevance[i])
        selected_indices.append(best_idx)

    selected_indices.sort()

    # Step 6: Optional hard cap
    if max_features is not None and len(selected_indices) > max_features:
        # Keep top max_features by relevance
        selected_indices.sort(key=lambda i: relevance[i], reverse=True)
        selected_indices = sorted(selected_indices[:max_features])

    selected_names = [feature_names[i] for i in selected_indices]

    logger.info(
        "mRMR: %d → %d features (threshold=%.2f)",
        n_features, len(selected_indices), spearman_threshold,
    )
    return selected_indices, selected_names
