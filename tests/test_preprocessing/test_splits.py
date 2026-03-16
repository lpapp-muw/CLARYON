"""Tests for claryon.preprocessing.splits — CV split generation."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.preprocessing.splits import (
    auto_split,
    generate_group_kfold_splits,
    generate_holdout_split,
    generate_kfold_splits,
    generate_nested_cv_splits,
)


@pytest.fixture
def binary_labels():
    return np.array([0, 1] * 50)


@pytest.fixture
def multiclass_labels():
    return np.array([0, 1, 2] * 40)


def test_kfold_basic(binary_labels):
    splits = generate_kfold_splits(binary_labels, n_folds=5, seed=42)
    assert len(splits) == 5
    # Every sample should appear in exactly one test fold
    all_test = np.concatenate([s.test_idx for s in splits])
    assert len(all_test) == len(binary_labels)
    assert len(set(all_test)) == len(binary_labels)


def test_kfold_reproducible(binary_labels):
    s1 = generate_kfold_splits(binary_labels, n_folds=5, seed=42)
    s2 = generate_kfold_splits(binary_labels, n_folds=5, seed=42)
    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a.train_idx, b.train_idx)
        np.testing.assert_array_equal(a.test_idx, b.test_idx)


def test_kfold_different_seeds(binary_labels):
    s1 = generate_kfold_splits(binary_labels, n_folds=5, seed=42)
    s2 = generate_kfold_splits(binary_labels, n_folds=5, seed=123)
    # At least one fold should differ
    any_different = any(
        not np.array_equal(a.test_idx, b.test_idx)
        for a, b in zip(s1, s2)
    )
    assert any_different


def test_kfold_stratified(binary_labels):
    splits = generate_kfold_splits(binary_labels, n_folds=5, seed=42)
    for s in splits:
        # Each test fold should have approximately equal class distribution
        test_y = binary_labels[s.test_idx]
        ratio = test_y.mean()
        assert 0.3 < ratio < 0.7


def test_holdout_split(binary_labels):
    splits = generate_holdout_split(binary_labels, test_size=0.2, seed=42)
    assert len(splits) == 1
    s = splits[0]
    assert len(s.test_idx) == pytest.approx(20, abs=2)
    assert len(s.train_idx) == pytest.approx(80, abs=2)


def test_group_kfold():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    splits = generate_group_kfold_splits(y, groups, n_folds=5)
    assert len(splits) == 5
    # Groups should not cross folds
    for s in splits:
        train_groups = set(groups[s.train_idx])
        test_groups = set(groups[s.test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_nested_cv(binary_labels):
    nested = generate_nested_cv_splits(binary_labels, outer_folds=3, inner_folds=2, seed=42)
    assert len(nested) == 3
    for outer_split, inner_splits in nested:
        assert len(inner_splits) == 2
        # Inner splits should use indices from outer training set only
        outer_train_set = set(outer_split.train_idx)
        for inner in inner_splits:
            assert set(inner.train_idx).issubset(outer_train_set)
            assert set(inner.test_idx).issubset(outer_train_set)


def test_auto_split_kfold(binary_labels):
    splits = auto_split(binary_labels, strategy="kfold", n_folds=5, seed=42)
    assert len(splits) == 5


def test_auto_split_large_dataset():
    y = np.array([0, 1] * 6000)  # 12000 samples > 10000 threshold
    splits = auto_split(y, strategy="kfold", n_folds=5, seed=42)
    # Should fall back to holdout for large dataset
    assert len(splits) == 1


def test_auto_split_holdout(binary_labels):
    splits = auto_split(binary_labels, strategy="holdout", seed=42)
    assert len(splits) == 1


def test_auto_split_unknown_strategy(binary_labels):
    with pytest.raises(ValueError, match="Unknown split strategy"):
        auto_split(binary_labels, strategy="invalid")
