"""Tests for SCST and LOCO split strategies."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.preprocessing.splits import (
    auto_split,
    generate_loco_splits,
    generate_scst_splits,
)


@pytest.fixture
def multicenter_data():
    """3 centers: A (10 samples), B (20 samples), C (5 samples)."""
    center_ids = np.array(
        ["A"] * 10 + ["B"] * 20 + ["C"] * 5
    )
    y = np.array([0, 1] * 5 + [0, 1] * 10 + [0, 1, 0, 1, 0])
    return y, center_ids


# ── SCST tests ──────────────────────────────────────────────


def test_scst_basic(multicenter_data):
    """3 centers → 6 splits."""
    y, center_ids = multicenter_data
    splits = generate_scst_splits(y, center_ids)
    assert len(splits) == 6


def test_scst_no_leakage(multicenter_data):
    """No sample appears in both train and test within any split."""
    y, center_ids = multicenter_data
    splits = generate_scst_splits(y, center_ids)
    for s in splits:
        train_set = set(s.train_idx)
        test_set = set(s.test_idx)
        assert train_set.isdisjoint(test_set)


def test_scst_center_isolation(multicenter_data):
    """All train samples from one center, all test from another."""
    y, center_ids = multicenter_data
    splits = generate_scst_splits(y, center_ids)
    for s in splits:
        train_centers = set(center_ids[s.train_idx])
        test_centers = set(center_ids[s.test_idx])
        assert len(train_centers) == 1, f"Train has multiple centers: {train_centers}"
        assert len(test_centers) == 1, f"Test has multiple centers: {test_centers}"
        assert train_centers.isdisjoint(test_centers)


def test_scst_all_centers_tested(multicenter_data):
    """Each center appears as test exactly N-1 times (2 for 3 centers)."""
    y, center_ids = multicenter_data
    splits = generate_scst_splits(y, center_ids)
    from collections import Counter

    test_center_counts = Counter()
    train_center_counts = Counter()
    for s in splits:
        test_c = center_ids[s.test_idx[0]]
        train_c = center_ids[s.train_idx[0]]
        test_center_counts[test_c] += 1
        train_center_counts[train_c] += 1

    for center in ["A", "B", "C"]:
        assert test_center_counts[center] == 2
        assert train_center_counts[center] == 2


def test_scst_reproducible(multicenter_data):
    """Same inputs produce identical splits."""
    y, center_ids = multicenter_data
    s1 = generate_scst_splits(y, center_ids)
    s2 = generate_scst_splits(y, center_ids)
    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a.train_idx, b.train_idx)
        np.testing.assert_array_equal(a.test_idx, b.test_idx)


# ── LOCO tests ──────────────────────────────────────────────


def test_loco_basic(multicenter_data):
    """3 centers → 3 splits."""
    y, center_ids = multicenter_data
    splits = generate_loco_splits(y, center_ids)
    assert len(splits) == 3


def test_loco_no_leakage(multicenter_data):
    """No center appears in both train and test."""
    y, center_ids = multicenter_data
    splits = generate_loco_splits(y, center_ids)
    for s in splits:
        train_centers = set(center_ids[s.train_idx])
        test_centers = set(center_ids[s.test_idx])
        assert train_centers.isdisjoint(test_centers)


def test_loco_all_centers_held_out(multicenter_data):
    """Each center held out exactly once."""
    y, center_ids = multicenter_data
    splits = generate_loco_splits(y, center_ids)
    held_out = [center_ids[s.test_idx[0]] for s in splits]
    assert sorted(held_out) == ["A", "B", "C"]


def test_loco_train_merges_remaining(multicenter_data):
    """Train set contains all non-test centers."""
    y, center_ids = multicenter_data
    splits = generate_loco_splits(y, center_ids)
    all_centers = set(np.unique(center_ids))
    for s in splits:
        test_center = center_ids[s.test_idx[0]]
        train_centers = set(center_ids[s.train_idx])
        assert train_centers == all_centers - {test_center}


# ── auto_split dispatch tests ───────────────────────────────


def test_auto_split_scst(multicenter_data):
    """auto_split dispatches to SCST correctly."""
    y, center_ids = multicenter_data
    splits = auto_split(y, strategy="scst", center_ids=center_ids)
    assert len(splits) == 6


def test_auto_split_loco(multicenter_data):
    """auto_split dispatches to LOCO correctly."""
    y, center_ids = multicenter_data
    splits = auto_split(y, strategy="loco", center_ids=center_ids)
    assert len(splits) == 3


def test_scst_missing_center_ids():
    """auto_split with scst and no center_ids raises ValueError."""
    y = np.array([0, 1] * 10)
    with pytest.raises(ValueError, match="center_ids"):
        auto_split(y, strategy="scst")


def test_loco_missing_center_ids():
    """auto_split with loco and no center_ids raises ValueError."""
    y = np.array([0, 1] * 10)
    with pytest.raises(ValueError, match="center_ids"):
        auto_split(y, strategy="loco")
