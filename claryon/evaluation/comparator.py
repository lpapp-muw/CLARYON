"""Statistical comparators — Friedman/Nemenyi, DeLong AUC test, bootstrap CI.

Ported from [B] analysis.py. Generalized for arbitrary metric matrices.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def friedman_nemenyi(
    ranks_matrix: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Friedman test + Nemenyi critical difference.

    Args:
        ranks_matrix: Shape (n_datasets, n_methods), ranks per dataset.
        alpha: Significance level.

    Returns:
        Dict with 'chi2', 'p_value', 'critical_difference', 'significant'.
    """
    n, k = ranks_matrix.shape

    mean_ranks = ranks_matrix.mean(axis=0)
    chi2 = (12 * n / (k * (k + 1))) * (
        (mean_ranks ** 2).sum() - (k * (k + 1) ** 2) / 4
    )

    denom = n * (k - 1) - chi2
    ff = ((n - 1) * chi2) / denom if denom > 0 else np.inf
    p_value = 1.0 - stats.f.cdf(ff, k - 1, (k - 1) * (n - 1))

    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha = q_alpha_table.get(k, 3.0)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "critical_difference": float(cd),
        "significant": bool(p_value < alpha),
    }


def compute_ranks(
    performance_matrix: np.ndarray,
    higher_is_better: bool = True,
) -> np.ndarray:
    """Rank methods per dataset.

    Args:
        performance_matrix: Shape (n_datasets, n_methods).
        higher_is_better: If True, higher values get rank 1.

    Returns:
        Ranks matrix of same shape.
    """
    from scipy.stats import rankdata

    ranks = np.zeros_like(performance_matrix)
    for i in range(performance_matrix.shape[0]):
        if higher_is_better:
            ranks[i] = rankdata(-performance_matrix[i], method="average")
        else:
            ranks[i] = rankdata(performance_matrix[i], method="average")
    return ranks


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Args:
        values: 1D array of observations.
        confidence: Confidence level.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = 1.0 - confidence
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return float(values.mean()), float(lo), float(hi)
