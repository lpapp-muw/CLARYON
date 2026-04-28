"""Integration tests for the multi-center domain shift study pipeline.

Verifies the full chain: SCST/LOCO splits → Hilbert flatten → Nyúl normalize
→ domain shift analysis → model training → bootstrap CI.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from claryon.evaluation.comparator import bootstrap_metric_ci
from claryon.evaluation.domain_shift import (
    center_classifier_bacc,
    mmd_pairwise,
)
from claryon.io.hilbert import flatten_volume
from claryon.preprocessing.image_prep import nyul_fit, nyul_transform
from claryon.preprocessing.splits import generate_loco_splits, generate_scst_splits
from tests.fixtures.synthetic_multicenter import generate_multicenter_volumes


@pytest.fixture
def multicenter_data():
    """20 samples per center, 16³ volumes."""
    return generate_multicenter_volumes(n_per_center=20, shape=(16, 16, 16), seed=42)


def test_scst_splits_on_synthetic(multicenter_data):
    """SCST produces 6 splits with correct center isolation."""
    splits = generate_scst_splits(
        multicenter_data["labels"],
        multicenter_data["center_ids"],
    )
    assert len(splits) == 6
    for s in splits:
        train_centers = set(multicenter_data["center_ids"][s.train_idx])
        test_centers = set(multicenter_data["center_ids"][s.test_idx])
        assert len(train_centers) == 1
        assert len(test_centers) == 1
        assert train_centers.isdisjoint(test_centers)


def test_loco_splits_on_synthetic(multicenter_data):
    """LOCO produces 3 splits, each with single held-out center."""
    splits = generate_loco_splits(
        multicenter_data["labels"],
        multicenter_data["center_ids"],
    )
    assert len(splits) == 3
    for s in splits:
        test_centers = set(multicenter_data["center_ids"][s.test_idx])
        assert len(test_centers) == 1


def test_hilbert_on_synthetic(multicenter_data):
    """Hilbert and rowmajor produce same-shaped output with equal sums."""
    vol = multicenter_data["volumes"][0]
    flat_rm = flatten_volume(vol, "rowmajor")
    flat_hb = flatten_volume(vol, "hilbert")
    assert flat_rm.shape == flat_hb.shape == (16 ** 3,)
    assert np.isclose(flat_rm.sum(), flat_hb.sum())


def test_nyul_on_synthetic(multicenter_data):
    """Fit Nyúl on DE+CH, transform AT volume without error."""
    vols = multicenter_data["volumes"]
    masks = multicenter_data["masks"]
    cids = multicenter_data["center_ids"]

    de_ch_idx = np.where((cids == "DE") | (cids == "CH"))[0]
    at_idx = np.where(cids == "AT")[0]

    landmarks = nyul_fit(
        [vols[i] for i in de_ch_idx],
        [masks[i] for i in de_ch_idx],
    )

    at_vol = vols[at_idx[0]]
    at_mask = masks[at_idx[0]]
    at_transformed = nyul_transform(at_vol, landmarks, mask=at_mask)

    assert at_transformed.shape == at_vol.shape
    # Unmasked voxels remain zero
    assert np.all(at_transformed[at_mask == 0] == 0)


def test_domain_shift_detects_synthetic(multicenter_data):
    """Center classifier BACC above chance on shifted synthetic data."""
    vols = multicenter_data["volumes"]
    flat = np.array([v.ravel() for v in vols])
    cids = multicenter_data["center_ids"]

    bacc_mean, _ = center_classifier_bacc(flat, cids)
    assert bacc_mean > 0.40


def test_mmd_on_synthetic(multicenter_data):
    """MMD pairwise produces non-negative values for all center pairs."""
    vols = multicenter_data["volumes"]
    flat = np.array([v.ravel() for v in vols])
    cids = multicenter_data["center_ids"]

    mmd_results = mmd_pairwise(flat, cids)
    # 3 unique pairs + 3 symmetric = 6 entries
    assert len(mmd_results) == 6
    assert all(v >= 0 for v in mmd_results.values())


def test_micro_cnn_on_synthetic(multicenter_data):
    """Train micro-CNN on one center, predict on another without crash."""
    pytest.importorskip("torch")
    from claryon.io.base import TaskType
    from claryon.models.classical.cnn_3d_micro import CNN3DMicroModel

    vols = multicenter_data["volumes"]
    labels = multicenter_data["labels"]
    cids = multicenter_data["center_ids"]

    train_idx = np.where(cids == "DE")[0]
    test_idx = np.where(cids == "CH")[0]

    X_train = vols[train_idx][:, np.newaxis, ...].astype(np.float32)
    X_test = vols[test_idx][:, np.newaxis, ...].astype(np.float32)
    y_train = labels[train_idx]

    model = CNN3DMicroModel(n_classes=2, epochs=2, seed=42)
    model.fit(X_train, y_train, TaskType.BINARY)
    probs = model.predict_proba(X_test)

    assert probs.shape == (len(test_idx), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_bootstrap_on_predictions(multicenter_data):
    """Bootstrap metric CI on synthetic predictions returns valid interval."""
    y_true = multicenter_data["labels"][:20]
    y_pred = y_true.copy()
    y_pred[:5] = 1 - y_pred[:5]

    def bacc(yt, yp):
        return balanced_accuracy_score(yt, yp)

    mean, lo, hi = bootstrap_metric_ci(y_true, y_pred, bacc)
    assert 0.3 < lo < mean < hi <= 1.0
