"""Tests for claryon.registry."""
from __future__ import annotations

import pytest

from claryon.registry import clear, get, list_registered, register


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with a clean registry."""
    clear()
    yield
    clear()


def test_register_and_get():
    @register("model", "dummy")
    class DummyModel:
        pass

    assert get("model", "dummy") is DummyModel


def test_duplicate_raises():
    @register("model", "dup")
    class First:
        pass

    with pytest.raises(ValueError, match="Duplicate"):
        @register("model", "dup")
        class Second:
            pass


def test_get_missing_raises():
    with pytest.raises(KeyError, match="Nothing registered"):
        get("model", "nonexistent")


def test_list_registered_filtered():
    @register("model", "a")
    class A:
        pass

    @register("metric", "b")
    def b():
        pass

    models = list_registered("model")
    assert "a" in models
    assert "b" not in models

    metrics = list_registered("metric")
    assert "b" in metrics


def test_list_registered_all():
    @register("model", "x")
    class X:
        pass

    @register("metric", "y")
    def y():
        pass

    all_items = list_registered()
    assert "model/x" in all_items
    assert "metric/y" in all_items


def test_clear_namespace():
    @register("model", "m1")
    class M1:
        pass

    @register("metric", "met1")
    def met1():
        pass

    clear("model")
    with pytest.raises(KeyError):
        get("model", "m1")
    # metric still exists
    assert get("metric", "met1") is met1


def test_register_function():
    @register("metric", "accuracy")
    def accuracy(y_true, y_pred):
        return 1.0

    assert get("metric", "accuracy") is accuracy
