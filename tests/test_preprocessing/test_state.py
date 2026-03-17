"""Tests for PreprocessingState — round-trip save/load, apply consistency."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.preprocessing.state import PreprocessingState


def _make_state() -> PreprocessingState:
    """Create a sample PreprocessingState for testing."""
    return PreprocessingState(
        z_mean=np.array([1.0, 2.0, 3.0, 4.0]),
        z_std=np.array([0.5, 1.0, 1.5, 2.0]),
        selected_features=[0, 2],
        selected_feature_names=["feat_0", "feat_2"],
        spearman_threshold=0.8,
        image_norm_mode="per_image",
        image_norm_min=None,
        image_norm_max=None,
        n_features_original=4,
        n_features_selected=2,
    )


def test_round_trip_save_load(tmp_path):
    """Save and load should produce identical state."""
    state = _make_state()
    path = tmp_path / "preprocessing_state.json"
    state.save(path)

    loaded = PreprocessingState.load(path)

    np.testing.assert_array_almost_equal(loaded.z_mean, state.z_mean)
    np.testing.assert_array_almost_equal(loaded.z_std, state.z_std)
    assert loaded.selected_features == state.selected_features
    assert loaded.selected_feature_names == state.selected_feature_names
    assert loaded.spearman_threshold == state.spearman_threshold
    assert loaded.image_norm_mode == state.image_norm_mode
    assert loaded.n_features_original == state.n_features_original
    assert loaded.n_features_selected == state.n_features_selected


def test_apply_tabular():
    """apply_tabular should z-score normalize and select features."""
    state = _make_state()
    X = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])

    result = state.apply_tabular(X)

    # Should select features 0 and 2 only
    assert result.shape == (2, 2)

    # Manual z-score for feature 0: (1 - 1) / 0.5 = 0, (2 - 1) / 0.5 = 2
    np.testing.assert_almost_equal(result[0, 0], 0.0)
    np.testing.assert_almost_equal(result[1, 0], 2.0)

    # Manual z-score for feature 2: (3 - 3) / 1.5 = 0, (6 - 3) / 1.5 = 2
    np.testing.assert_almost_equal(result[0, 1], 0.0)
    np.testing.assert_almost_equal(result[1, 1], 2.0)


def test_apply_tabular_consistency_with_load(tmp_path):
    """apply_tabular should give same result before and after save/load."""
    state = _make_state()
    X = np.array([[1.5, 2.5, 3.5, 4.5]])

    result_before = state.apply_tabular(X)

    path = tmp_path / "state.json"
    state.save(path)
    loaded = PreprocessingState.load(path)
    result_after = loaded.apply_tabular(X)

    np.testing.assert_array_almost_equal(result_before, result_after)


def test_apply_image_per_image():
    """per_image mode should scale each volume independently to [0, 1]."""
    state = PreprocessingState(
        z_mean=np.array([0.0]),
        z_std=np.array([1.0]),
        selected_features=[0],
        selected_feature_names=["f0"],
        spearman_threshold=0.8,
        image_norm_mode="per_image",
    )
    volumes = np.array([[[0.0, 10.0], [5.0, 20.0]],
                        [[100.0, 200.0], [150.0, 300.0]]])  # shape (2, 2, 2)
    result = state.apply_image(volumes)

    # Each volume should be [0, 1]
    for i in range(result.shape[0]):
        assert result[i].min() == pytest.approx(0.0)
        assert result[i].max() == pytest.approx(1.0)


def test_apply_image_cohort_global():
    """cohort_global mode should use stored min/max from training."""
    state = PreprocessingState(
        z_mean=np.array([0.0]),
        z_std=np.array([1.0]),
        selected_features=[0],
        selected_feature_names=["f0"],
        spearman_threshold=0.8,
        image_norm_mode="cohort_global",
        image_norm_min=0.0,
        image_norm_max=100.0,
    )
    volumes = np.array([[[50.0, 100.0], [0.0, 150.0]]])
    result = state.apply_image(volumes)

    np.testing.assert_almost_equal(result[0, 0, 0], 0.5)
    np.testing.assert_almost_equal(result[0, 0, 1], 1.0)
    np.testing.assert_almost_equal(result[0, 1, 0], 0.0)
    # 150 clipped to 1.0
    np.testing.assert_almost_equal(result[0, 1, 1], 1.0)
