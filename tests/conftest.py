"""Shared pytest fixtures for the CLARYON test suite.

All fixtures load from pre-generated data in tests/fixtures/data/.
No data is regenerated during test runs.

Usage in test files:
    def test_something(tabular_binary_train, synthetic_nifti_masked):
        ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

# ── Paths ──

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "data"


def _require_fixture(path: Path) -> Path:
    """Assert fixture path exists, with helpful error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture not found: {path}\n"
            f"Run: python tests/fixtures/generate_fixtures.py"
        )
    return path


# ═══════════════════════════════════════════════════════════════
# Tabular fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def tabular_binary_dir() -> Path:
    return _require_fixture(FIXTURES_DIR / "tabular_binary")


@pytest.fixture(scope="session")
def tabular_binary_train(tabular_binary_dir: Path) -> pd.DataFrame:
    return pd.read_csv(tabular_binary_dir / "train.csv", sep=";")


@pytest.fixture(scope="session")
def tabular_binary_test(tabular_binary_dir: Path) -> pd.DataFrame:
    return pd.read_csv(tabular_binary_dir / "test.csv", sep=";")


@pytest.fixture(scope="session")
def tabular_binary_Xy_train(tabular_binary_train: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = tabular_binary_train.drop(columns=["label"]).values
    y = tabular_binary_train["label"].values
    return X, y


@pytest.fixture(scope="session")
def tabular_binary_Xy_test(tabular_binary_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = tabular_binary_test.drop(columns=["label"]).values
    y = tabular_binary_test["label"].values
    return X, y


@pytest.fixture(scope="session")
def tabular_multiclass_dir() -> Path:
    return _require_fixture(FIXTURES_DIR / "tabular_multiclass")


@pytest.fixture(scope="session")
def tabular_multiclass_train(tabular_multiclass_dir: Path) -> pd.DataFrame:
    return pd.read_csv(tabular_multiclass_dir / "train.csv", sep=";")


@pytest.fixture(scope="session")
def tabular_multiclass_test(tabular_multiclass_dir: Path) -> pd.DataFrame:
    return pd.read_csv(tabular_multiclass_dir / "test.csv", sep=";")


@pytest.fixture(scope="session")
def tabular_regression_dir() -> Path:
    return _require_fixture(FIXTURES_DIR / "tabular_regression")


@pytest.fixture(scope="session")
def tabular_regression_train(tabular_regression_dir: Path) -> pd.DataFrame:
    return pd.read_csv(tabular_regression_dir / "train.csv", sep=";")


@pytest.fixture(scope="session")
def tabular_regression_test(tabular_regression_dir: Path) -> pd.DataFrame:
    return pd.read_csv(tabular_regression_dir / "test.csv", sep=";")


# ═══════════════════════════════════════════════════════════════
# NIfTI fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def synthetic_nifti_masked() -> Path:
    """Path to NIfTI dataset with binary masks (Train/ + Test/)."""
    return _require_fixture(FIXTURES_DIR / "nifti_masked")


@pytest.fixture(scope="session")
def synthetic_nifti_nomask() -> Path:
    """Path to NIfTI dataset without masks (Train/ + Test/)."""
    return _require_fixture(FIXTURES_DIR / "nifti_nomask")


@pytest.fixture(scope="session")
def synthetic_nifti_multilabel() -> Path:
    """Path to NIfTI dataset with integer multi-label masks."""
    return _require_fixture(FIXTURES_DIR / "nifti_multilabel")


@pytest.fixture(scope="session")
def nifti_image_mask_pairs(synthetic_nifti_masked: Path) -> List[Tuple[Path, Path]]:
    """List of (image_path, mask_path) tuples from the masked NIfTI fixture."""
    train_dir = synthetic_nifti_masked / "Train"
    pairs = []
    pet_files = sorted(train_dir.glob("*_PET.nii.gz"))
    for pet in pet_files:
        mask_name = pet.name.replace("_PET.nii.gz", "_mask.nii.gz")
        mask = pet.parent / mask_name
        if mask.exists():
            pairs.append((pet, mask))
    return pairs


# ═══════════════════════════════════════════════════════════════
# FDB / LDB fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def fdb_ldb_dir() -> Path:
    return _require_fixture(FIXTURES_DIR / "fdb_ldb")


@pytest.fixture(scope="session")
def fdb_path(fdb_ldb_dir: Path) -> Path:
    return fdb_ldb_dir / "FDB.csv"


@pytest.fixture(scope="session")
def ldb_path(fdb_ldb_dir: Path) -> Path:
    return fdb_ldb_dir / "LDB.csv"


# ═══════════════════════════════════════════════════════════════
# TIFF fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def tiff_dir() -> Path:
    return _require_fixture(FIXTURES_DIR / "tiff_synthetic")


# ═══════════════════════════════════════════════════════════════
# PyRadiomics config
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def pyradiomics_config() -> Path:
    return _require_fixture(FIXTURES_DIR / "pyradiomics_minimal.yaml")


# ═══════════════════════════════════════════════════════════════
# Quantum / amplitude encoding fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def synthetic_amplitude_data() -> Dict[str, Any]:
    """Small amplitude-encoded dataset for quantum model smoke tests.

    4 qubits → 16-dimensional vectors, 20 train + 10 test samples.
    Deterministic.
    """
    rng = np.random.default_rng(42)
    n_qubits = 4
    dim = 2 ** n_qubits  # 16

    def _make_encoded(n: int) -> Tuple[np.ndarray, np.ndarray]:
        X_raw = rng.standard_normal((n, dim))
        # L2-normalize each row (amplitude encoding constraint)
        norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_amp = X_raw / norms
        y = (X_raw[:, 0] > 0).astype(int)
        return X_amp, y

    X_train, y_train = _make_encoded(20)
    X_test, y_test = _make_encoded(10)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "n_qubits": n_qubits,
        "pad_len": dim,
    }


# ═══════════════════════════════════════════════════════════════
# CNN / imaging fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def synthetic_2d_images() -> Dict[str, Any]:
    """Small 2D image dataset for CNN smoke tests.

    20 train + 10 test images, 1 channel, 32x32 pixels, binary labels.
    """
    rng = np.random.default_rng(55)

    def _make(n: int) -> Tuple[np.ndarray, np.ndarray]:
        X = rng.standard_normal((n, 1, 32, 32)).astype(np.float32)
        y = np.array([i % 2 for i in range(n)], dtype=int)
        # Add signal for class 1
        for i in range(n):
            if y[i] == 1:
                X[i, 0, 10:20, 10:20] += 2.0
        return X, y

    X_train, y_train = _make(20)
    X_test, y_test = _make(10)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture(scope="session")
def synthetic_3d_volumes() -> Dict[str, Any]:
    """Small 3D volume dataset for 3D CNN smoke tests.

    10 train + 5 test volumes, 1 channel, 10x12x8 voxels, binary labels.
    """
    rng = np.random.default_rng(66)

    def _make(n: int) -> Tuple[np.ndarray, np.ndarray]:
        X = rng.standard_normal((n, 1, 10, 12, 8)).astype(np.float32)
        y = np.array([i % 2 for i in range(n)], dtype=int)
        for i in range(n):
            if y[i] == 1:
                X[i, 0, 3:7, 3:9, 2:6] += 2.0
        return X, y

    X_train, y_train = _make(10)
    X_test, y_test = _make(5)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


# ═══════════════════════════════════════════════════════════════
# Temporary directory fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for tests that write files."""
    d = tmp_path / "output"
    d.mkdir()
    return d
