"""Tests for L2-normalized CNN and micro-CNN variants."""
from __future__ import annotations

import numpy as np
import pytest

from claryon.io.base import TaskType


@pytest.fixture
def vol_data_32():
    """Small 32³ volume dataset for variant CNN tests."""
    rng = np.random.default_rng(42)
    X = rng.random((4, 1, 32, 32, 32)).astype(np.float32)
    y = np.array([0, 1, 0, 1])
    return X, y


# ── L2-normalized CNN tests ─────────────────────────────────


def test_cnn_3d_l2_instantiate():
    pytest.importorskip("torch")
    from claryon.models.classical.cnn_3d_l2 import CNN3DL2Model

    model = CNN3DL2Model(n_classes=2, epochs=1)
    assert model.name == "cnn_3d_l2"


def test_cnn_3d_l2_forward(vol_data_32):
    pytest.importorskip("torch")
    from claryon.models.classical.cnn_3d_l2 import CNN3DL2Model

    X, y = vol_data_32
    model = CNN3DL2Model(n_classes=2, epochs=1, seed=42)
    model.fit(X, y, TaskType.BINARY)
    probs = model.predict_proba(X)
    assert probs.shape == (4, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_cnn_3d_l2_differs_from_base(vol_data_32):
    """Same data, same seed → L2 model output differs from base."""
    pytest.importorskip("torch")
    from claryon.models.classical.cnn_3d import CNN3DModel
    from claryon.models.classical.cnn_3d_l2 import CNN3DL2Model

    X, y = vol_data_32
    m_base = CNN3DModel(n_classes=2, epochs=2, seed=42)
    m_l2 = CNN3DL2Model(n_classes=2, epochs=2, seed=42)
    m_base.fit(X, y, TaskType.BINARY)
    m_l2.fit(X, y, TaskType.BINARY)
    p_base = m_base.predict_proba(X)
    p_l2 = m_l2.predict_proba(X)
    assert not np.allclose(p_base, p_l2)


def test_cnn_3d_l2_registered():
    pytest.importorskip("torch")
    from claryon.models.classical.cnn_3d_l2 import CNN3DL2Model
    from claryon.registry import get

    assert get("model", "cnn_3d_l2") is CNN3DL2Model


# ── Micro-CNN tests ──────────────────────────────────────────


def test_cnn_3d_micro_instantiate():
    pytest.importorskip("torch")
    from claryon.models.classical.cnn_3d_micro import CNN3DMicroModel

    model = CNN3DMicroModel(n_classes=2, epochs=1)
    assert model.name == "cnn_3d_micro"


def test_cnn_3d_micro_forward(vol_data_32):
    pytest.importorskip("torch")
    from claryon.models.classical.cnn_3d_micro import CNN3DMicroModel

    X, y = vol_data_32
    model = CNN3DMicroModel(n_classes=2, epochs=1, seed=42)
    model.fit(X, y, TaskType.BINARY)
    probs = model.predict_proba(X)
    assert probs.shape == (4, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_cnn_3d_micro_param_count(vol_data_32):
    pytest.importorskip("torch")
    from claryon.models.classical.cnn_3d_micro import CNN3DMicroModel

    X, y = vol_data_32
    model = CNN3DMicroModel(n_classes=2, epochs=1, seed=42)
    model.fit(X, y, TaskType.BINARY)
    n_params = sum(p.numel() for p in model._model.parameters())
    assert 3_000 <= n_params <= 15_000, f"Got {n_params} params"


def test_cnn_3d_micro_registered():
    pytest.importorskip("torch")
    from claryon.models.classical.cnn_3d_micro import CNN3DMicroModel
    from claryon.registry import get

    assert get("model", "cnn_3d_micro") is CNN3DMicroModel


def test_cnn_3d_micro_works_with_16_cube():
    """Non-32 cube size (future datasets may use different sizes)."""
    pytest.importorskip("torch")
    from claryon.models.classical.cnn_3d_micro import CNN3DMicroModel

    rng = np.random.default_rng(42)
    X = rng.random((2, 1, 16, 16, 16)).astype(np.float32)
    y = np.array([0, 1])
    model = CNN3DMicroModel(n_classes=2, epochs=1, seed=42)
    model.fit(X, y, TaskType.BINARY)
    probs = model.predict_proba(X)
    assert probs.shape == (2, 2)
