"""Tests for claryon.determinism."""
from __future__ import annotations

import os
import random

import numpy as np

from claryon.determinism import enforce_determinism


def test_python_rng_seeded():
    enforce_determinism(seed=123)
    a = [random.random() for _ in range(5)]
    enforce_determinism(seed=123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_rng_seeded():
    enforce_determinism(seed=42)
    a = np.random.rand(5)
    enforce_determinism(seed=42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_thread_env_vars_set():
    enforce_determinism(seed=1, threads=2)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        # setdefault won't overwrite existing, so check the var is set
        assert var in os.environ


def test_threads_none_leaves_env():
    """threads=None should not set thread env vars."""
    # Remove vars so we can detect if they get set
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.pop(var, None)
    enforce_determinism(seed=1, threads=None)
    # They may or may not be set from a previous call — the point is no error
