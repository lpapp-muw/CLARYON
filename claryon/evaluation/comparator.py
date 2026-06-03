"""Statistical comparators — Friedman/Nemenyi, DeLong AUC test, bootstrap CI.

Ported from [B] analysis.py. Generalized for arbitrary metric matrices.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

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


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for a metric computed on (y_true, y_pred) pairs.

    Resamples paired (y_true[i], y_pred[i]) with replacement, computes
    the metric on each resample, and returns percentile-based CI.

    Args:
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted labels, shape (n_samples,).
        metric_fn: Callable(y_true, y_pred) -> float.
        n_bootstrap: Number of resamples.
        confidence: CI level.
        seed: Random seed.

    Returns:
        Tuple of (point_estimate, lower, upper).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    point = metric_fn(y_true, y_pred)

    boot_values = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_values[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = 1.0 - confidence
    lo = np.percentile(boot_values, 100 * alpha / 2)
    hi = np.percentile(boot_values, 100 * (1 - alpha / 2))

    return float(point), float(lo), float(hi)


def paired_superiority(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> float:
    """Probability that method A outperforms method B.

    Bootstrap resamples (y_true, preds_a, preds_b) with the same indices
    and computes P(metric(A) > metric(B)) across resamples.

    Args:
        y_true: True labels, shape (n_samples,).
        preds_a: Predictions from method A, shape (n_samples,).
        preds_b: Predictions from method B, shape (n_samples,).
        metric_fn: Callable(y_true, y_pred) -> float. Higher is better.
        n_bootstrap: Number of resamples.
        seed: Random seed.

    Returns:
        Fraction of resamples where metric(A) > metric(B).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    n = len(y_true)

    a_wins = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        score_a = metric_fn(y_true[idx], preds_a[idx])
        score_b = metric_fn(y_true[idx], preds_b[idx])
        if score_a > score_b:
            a_wins += 1

    return a_wins / n_bootstrap


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> Tuple[float, float]:
    """McNemar's test for paired binary classifiers.

    Computes the 2x2 discordance table:
      b = count(A correct & B wrong)
      c = count(A wrong & B correct)
    Uses exact binomial test if b+c < 25, otherwise chi-squared
    with continuity correction.

    Args:
        y_true: True labels, shape (n_samples,).
        preds_a: Predictions from method A, shape (n_samples,).
        preds_b: Predictions from method B, shape (n_samples,).

    Returns:
        Tuple of (statistic, p_value).
    """
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    a_correct = preds_a == y_true
    b_correct = preds_b == y_true

    b = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
    c = int(np.sum(~a_correct & b_correct))  # A wrong, B right

    if b + c == 0:
        return 0.0, 1.0

    if b + c < 25:
        # Exact binomial test
        result = stats.binomtest(b, b + c, 0.5)
        return float(b), float(result.pvalue)

    # Chi-squared with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1.0 - stats.chi2.cdf(statistic, df=1)
    return float(statistic), float(p_value)
