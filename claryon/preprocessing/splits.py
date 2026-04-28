"""Cross-validation split generation — k-fold, holdout, nested CV, GroupKFold, SCST, LOCO.

Ported from [B] fold_generator.py. Generalized with nested CV, GroupKFold,
and center-aware strategies (SCST, LOCO) for multi-center studies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

logger = logging.getLogger(__name__)

LARGE_DATASET_THRESHOLD = 10000


@dataclass
class SplitIndices:
    """Train/test index pair for a single fold.

    Attributes:
        train_idx: Array of training sample indices.
        test_idx: Array of test sample indices.
        fold: Fold number.
        seed: Random seed used.
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    fold: int
    seed: int


def generate_kfold_splits(
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> List[SplitIndices]:
    """Generate stratified k-fold splits.

    Args:
        y: Label array for stratification.
        n_folds: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        List of SplitIndices, one per fold.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(y), 1))
    splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_dummy, y)):
        splits.append(SplitIndices(
            train_idx=train_idx, test_idx=test_idx, fold=fold_idx, seed=seed,
        ))
    return splits


def generate_holdout_split(
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> List[SplitIndices]:
    """Generate a single stratified holdout split.

    Args:
        y: Label array for stratification.
        test_size: Fraction of data to use for test set.
        seed: Random seed for reproducibility.

    Returns:
        List with a single SplitIndices.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    X_dummy = np.zeros((len(y), 1))
    train_idx, test_idx = next(sss.split(X_dummy, y))
    return [SplitIndices(
        train_idx=train_idx, test_idx=test_idx, fold=0, seed=seed,
    )]


def generate_large_dataset_split(
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> List[SplitIndices]:
    """Generate a fixed 80/20 split for large datasets.

    Per HF-007: datasets with N > 10K use a fixed stratified split
    instead of k-fold.

    Args:
        y: Label array for stratification.
        test_size: Fraction for test set (default 0.2 → 80/20).
        seed: Random seed.

    Returns:
        List with a single SplitIndices.
    """
    return generate_holdout_split(y, test_size=test_size, seed=seed)


def generate_group_kfold_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
) -> List[SplitIndices]:
    """Generate GroupKFold splits (no stratification, groups don't cross folds).

    Args:
        y: Label array.
        groups: Group membership array (same length as y).
        n_folds: Number of folds.

    Returns:
        List of SplitIndices.
    """
    gkf = GroupKFold(n_splits=n_folds)
    X_dummy = np.zeros((len(y), 1))
    splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_dummy, y, groups)):
        splits.append(SplitIndices(
            train_idx=train_idx, test_idx=test_idx, fold=fold_idx, seed=0,
        ))
    return splits


def generate_nested_cv_splits(
    y: np.ndarray,
    outer_folds: int = 5,
    inner_folds: int = 3,
    seed: int = 42,
) -> List[Tuple[SplitIndices, List[SplitIndices]]]:
    """Generate nested cross-validation splits.

    Outer loop for evaluation, inner loop for hyperparameter selection.

    Args:
        y: Label array for stratification.
        outer_folds: Number of outer folds.
        inner_folds: Number of inner folds.
        seed: Random seed.

    Returns:
        List of (outer_split, inner_splits) tuples. Each outer_split is a
        SplitIndices for the outer fold, and inner_splits is a list of
        SplitIndices for the inner folds (trained on outer's training set).
    """
    outer_skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(y), 1))

    result = []
    for outer_idx, (outer_train, outer_test) in enumerate(outer_skf.split(X_dummy, y)):
        outer_split = SplitIndices(
            train_idx=outer_train, test_idx=outer_test,
            fold=outer_idx, seed=seed,
        )

        # Inner splits on the outer training set
        y_inner = y[outer_train]
        inner_skf = StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=seed + outer_idx + 1,
        )
        X_inner = np.zeros((len(y_inner), 1))
        inner_splits = []
        for inner_idx, (rel_train, rel_test) in enumerate(inner_skf.split(X_inner, y_inner)):
            # Map relative indices back to absolute indices
            abs_train = outer_train[rel_train]
            abs_test = outer_train[rel_test]
            inner_splits.append(SplitIndices(
                train_idx=abs_train, test_idx=abs_test,
                fold=inner_idx, seed=seed + outer_idx + 1,
            ))

        result.append((outer_split, inner_splits))

    return result


def generate_scst_splits(
    y: np.ndarray,
    center_ids: np.ndarray,
) -> List[SplitIndices]:
    """Single-Center-Train, Single-Center-Test splits.

    For N centers, generates N*(N-1) splits. Each split trains on all
    samples from one center and tests on all samples from a different center.

    Args:
        y: Label array (used for consistency, not for stratification).
        center_ids: Array of center identifiers, same length as y.

    Returns:
        List of SplitIndices, one per (train_center, test_center) pair.
    """
    centers = sorted(np.unique(center_ids))
    logger.info("SCST: %d centers → %d splits", len(centers), len(centers) * (len(centers) - 1))

    splits = []
    fold = 0
    for train_center in centers:
        for test_center in centers:
            if train_center == test_center:
                continue
            train_idx = np.where(center_ids == train_center)[0]
            test_idx = np.where(center_ids == test_center)[0]
            splits.append(SplitIndices(
                train_idx=train_idx, test_idx=test_idx, fold=fold, seed=0,
            ))
            fold += 1
    return splits


def generate_loco_splits(
    y: np.ndarray,
    center_ids: np.ndarray,
) -> List[SplitIndices]:
    """Leave-One-Center-Out splits.

    For N centers, generates N splits. Each split trains on N-1 centers
    merged and tests on the held-out center.

    Args:
        y: Label array (used for consistency, not for stratification).
        center_ids: Array of center identifiers, same length as y.

    Returns:
        List of SplitIndices, one per held-out center.
    """
    centers = sorted(np.unique(center_ids))
    logger.info("LOCO: %d centers → %d splits", len(centers), len(centers))

    splits = []
    for fold, held_out in enumerate(centers):
        test_idx = np.where(center_ids == held_out)[0]
        train_idx = np.where(center_ids != held_out)[0]
        splits.append(SplitIndices(
            train_idx=train_idx, test_idx=test_idx, fold=fold, seed=0,
        ))
    return splits


def auto_split(
    y: np.ndarray,
    strategy: str = "kfold",
    n_folds: int = 5,
    seed: int = 42,
    test_size: float = 0.2,
    groups: Optional[np.ndarray] = None,
    center_ids: Optional[np.ndarray] = None,
    outer_folds: int = 5,
    inner_folds: int = 3,
    large_threshold: int = LARGE_DATASET_THRESHOLD,
) -> List[SplitIndices]:
    """Auto-select split strategy based on configuration and dataset size.

    For k-fold on large datasets (N > threshold), automatically falls back
    to a fixed holdout split per HF-007.

    Args:
        y: Label array.
        strategy: One of "kfold", "holdout", "nested", "group_kfold",
            "scst", "loco".
        n_folds: Number of folds for k-fold strategies.
        seed: Random seed.
        test_size: Test fraction for holdout.
        groups: Group array for group_kfold.
        center_ids: Center identifier array for scst/loco strategies.
        outer_folds: Outer folds for nested CV.
        inner_folds: Inner folds for nested CV.
        large_threshold: Sample count threshold for large dataset handling.

    Returns:
        List of SplitIndices.
    """
    n = len(y)

    if strategy == "kfold":
        if n > large_threshold:
            logger.info(
                "Dataset has %d samples (> %d threshold), using fixed holdout split",
                n, large_threshold,
            )
            return generate_large_dataset_split(y, test_size=test_size, seed=seed)
        return generate_kfold_splits(y, n_folds=n_folds, seed=seed)

    if strategy == "holdout":
        return generate_holdout_split(y, test_size=test_size, seed=seed)

    if strategy == "group_kfold":
        if groups is None:
            raise ValueError("groups array required for group_kfold strategy")
        return generate_group_kfold_splits(y, groups, n_folds=n_folds)

    if strategy == "nested":
        # Flatten nested structure to just outer splits for compatibility
        nested = generate_nested_cv_splits(y, outer_folds, inner_folds, seed)
        return [outer for outer, _ in nested]

    if strategy == "scst":
        if center_ids is None:
            raise ValueError("center_ids array required for scst strategy")
        return generate_scst_splits(y, center_ids)

    if strategy == "loco":
        if center_ids is None:
            raise ValueError("center_ids array required for loco strategy")
        return generate_loco_splits(y, center_ids)

    raise ValueError(f"Unknown split strategy: {strategy!r}")
