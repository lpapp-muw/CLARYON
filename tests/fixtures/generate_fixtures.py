#!/usr/bin/env python3
"""Generate all synthetic test fixtures for the CLARYON test suite.

Run once during project setup. Outputs are committed to tests/fixtures/data/.
Deterministic: same seed → same data every time.

Fixtures generated:
  - Tabular: binary, multi-class, regression (train + test splits)
  - NIfTI: masked, unmasked, multi-label mask
  - FDB/LDB: legacy radiomics format
  - TIFF: synthetic with metadata sidecar
  - PyRadiomics: minimal config YAML
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = Path(__file__).resolve().parent / "data"


def _rng(extra_seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + extra_seed)


# ═══════════════════════════════════════════════════════════════
# Tabular fixtures
# ═══════════════════════════════════════════════════════════════

def generate_tabular_binary() -> None:
    """80 train + 20 test samples, 10 features, binary label (0/1)."""
    out = DATA_DIR / "tabular_binary"
    out.mkdir(parents=True, exist_ok=True)
    rng = _rng(1)

    for split, n in [("train", 80), ("test", 20)]:
        X = rng.standard_normal((n, 10))
        # Label correlated with first two features
        logits = 1.5 * X[:, 0] - 1.0 * X[:, 1] + rng.standard_normal(n) * 0.3
        y = (logits > 0).astype(int)
        cols = [f"f{i}" for i in range(10)] + ["label"]
        df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
        df["label"] = df["label"].astype(int)
        df.to_csv(out / f"{split}.csv", sep=";", index=False)


def generate_tabular_multiclass() -> None:
    """120 train + 30 test samples, 10 features, 3 classes (0/1/2)."""
    out = DATA_DIR / "tabular_multiclass"
    out.mkdir(parents=True, exist_ok=True)
    rng = _rng(2)

    for split, n in [("train", 120), ("test", 30)]:
        X = rng.standard_normal((n, 10))
        # 3 clusters
        centers = np.array([[2, 0], [-1, 2], [-1, -2]], dtype=float)
        y = np.argmin(
            np.sum((X[:, :2][:, None, :] - centers[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        cols = [f"f{i}" for i in range(10)] + ["label"]
        df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
        df["label"] = df["label"].astype(int)
        df.to_csv(out / f"{split}.csv", sep=";", index=False)


def generate_tabular_regression() -> None:
    """80 train + 20 test samples, 10 features, continuous target."""
    out = DATA_DIR / "tabular_regression"
    out.mkdir(parents=True, exist_ok=True)
    rng = _rng(3)

    for split, n in [("train", 80), ("test", 20)]:
        X = rng.standard_normal((n, 10))
        y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2] + rng.standard_normal(n) * 0.5
        cols = [f"f{i}" for i in range(10)] + ["target"]
        df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
        df.to_csv(out / f"{split}.csv", sep=";", index=False)


# ═══════════════════════════════════════════════════════════════
# NIfTI fixtures
# ═══════════════════════════════════════════════════════════════

def _write_nifti(path: Path, arr: np.ndarray) -> None:
    """Write a NumPy array as a NIfTI .nii.gz file."""
    import nibabel as nib

    img = nib.Nifti1Image(arr.astype(np.float32), np.eye(4, dtype=np.float32))
    nib.save(img, str(path))


def _make_nifti_case(
    shape: tuple[int, ...], label: int, rng: np.random.Generator, with_mask: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Generate one synthetic PET volume (± mask)."""
    pet = rng.normal(loc=0.0, scale=0.2, size=shape).astype(np.float32)
    z0, y0, x0 = [s // 4 for s in shape]
    z1, y1, x1 = [s - s // 4 for s in shape]
    roi = np.zeros(shape, dtype=np.float32)
    roi[z0:z1, y0:y1, x0:x1] = 1.0

    if label == 1:
        pet += roi * 2.0

    return pet, roi.copy() if with_mask else None


def generate_nifti_masked() -> None:
    """10 train + 6 test NIfTI volumes with paired binary masks.

    Shape: (10, 12, 8). Labels in filename: case_<id>_<label>_PET.nii.gz
    """
    out = DATA_DIR / "nifti_masked"
    shape = (10, 12, 8)
    rng = _rng(10)

    for split, n in [("Train", 10), ("Test", 6)]:
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            label = i % 2
            pet, mask = _make_nifti_case(shape, label, rng, with_mask=True)
            case = f"case{split.lower()}{i:03d}_{label}"
            _write_nifti(d / f"{case}_PET.nii.gz", pet)
            _write_nifti(d / f"{case}_mask.nii.gz", mask)


def generate_nifti_nomask() -> None:
    """10 train + 6 test NIfTI volumes WITHOUT masks."""
    out = DATA_DIR / "nifti_nomask"
    shape = (10, 12, 8)
    rng = _rng(20)

    for split, n in [("Train", 10), ("Test", 6)]:
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            label = i % 2
            pet, _ = _make_nifti_case(shape, label, rng, with_mask=False)
            case = f"case{split.lower()}{i:03d}_{label}"
            _write_nifti(d / f"{case}_PET.nii.gz", pet)


def generate_nifti_multilabel() -> None:
    """5 volumes with integer-labelled masks (3 ROIs per volume).

    Tests multi-label mask support for pyradiomics extraction.
    """
    out = DATA_DIR / "nifti_multilabel"
    out.mkdir(parents=True, exist_ok=True)
    shape = (12, 12, 12)
    rng = _rng(30)

    for i in range(5):
        label = i % 2
        pet = rng.normal(loc=0.5, scale=0.3, size=shape).astype(np.float32)

        # Create 3 non-overlapping ROIs
        mask = np.zeros(shape, dtype=np.int16)
        mask[1:4, 1:4, 1:4] = 1  # ROI 1
        mask[5:8, 5:8, 5:8] = 2  # ROI 2
        mask[9:11, 9:11, 9:11] = 3  # ROI 3

        if label == 1:
            pet[mask == 1] += 1.5
            pet[mask == 2] += 2.0

        case = f"multilabel{i:03d}_{label}"
        _write_nifti(out / f"{case}_PET.nii.gz", pet)
        _write_nifti(out / f"{case}_mask.nii.gz", mask.astype(np.float32))


# ═══════════════════════════════════════════════════════════════
# FDB / LDB fixtures (legacy radiomics format)
# ═══════════════════════════════════════════════════════════════

def generate_fdb_ldb() -> None:
    """Small FDB (feature database) + LDB (label database) in semicolon format."""
    out = DATA_DIR / "fdb_ldb"
    out.mkdir(parents=True, exist_ok=True)
    rng = _rng(40)

    n = 30
    n_features = 8
    feature_names = [f"Modality::Group::feat{i}" for i in range(n_features)]

    # FDB: Key + features, semicolon-separated
    keys = [f"S{i:04d}" for i in range(n)]
    X = rng.standard_normal((n, n_features))
    header = "Key;" + ";".join(feature_names)
    lines = [header]
    for i in range(n):
        vals = ";".join(f"{v:.8f}" for v in X[i])
        lines.append(f"{keys[i]};{vals}")
    (out / "FDB.csv").write_text("\n".join(lines) + "\n")

    # LDB: Key + label (Low-High format)
    labels = ["Low" if rng.random() < 0.5 else "High" for _ in range(n)]
    ldb_lines = ["Key;Low-High"]
    for i in range(n):
        ldb_lines.append(f"{keys[i]};{labels[i]}")
    (out / "LDB.csv").write_text("\n".join(ldb_lines) + "\n")


# ═══════════════════════════════════════════════════════════════
# TIFF fixtures
# ═══════════════════════════════════════════════════════════════

def generate_tiff_synthetic() -> None:
    """5 synthetic TIFF images with JSON metadata sidecars."""
    import json

    out = DATA_DIR / "tiff_synthetic"
    out.mkdir(parents=True, exist_ok=True)
    rng = _rng(50)

    try:
        import tifffile
    except ImportError:
        # tifffile not installed — write raw numpy files as placeholder
        print("  WARNING: tifffile not installed; writing .npy placeholders for TIFF fixtures")
        for i in range(5):
            label = i % 2
            img = rng.standard_normal((64, 64)).astype(np.float32)
            if label == 1:
                img[20:40, 20:40] += 2.0
            np.save(out / f"sample_{i:03d}.npy", img)
            meta = {"label": label, "pixel_size_um": 5.0, "wavelength_nm": 850}
            (out / f"sample_{i:03d}.json").write_text(json.dumps(meta))
        return

    for i in range(5):
        label = i % 2
        img = rng.standard_normal((64, 64)).astype(np.float32)
        if label == 1:
            img[20:40, 20:40] += 2.0
        tifffile.imwrite(str(out / f"sample_{i:03d}.tif"), img)
        meta = {"label": label, "pixel_size_um": 5.0, "wavelength_nm": 850}
        (out / f"sample_{i:03d}.json").write_text(json.dumps(meta))


# ═══════════════════════════════════════════════════════════════
# PyRadiomics config
# ═══════════════════════════════════════════════════════════════

def generate_pyradiomics_config() -> None:
    """Minimal pyradiomics YAML: firstorder + glcm only, bin width 25.

    Designed for fast extraction on small synthetic volumes (~seconds per volume).
    """
    out = DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    config = """\
setting:
  binWidth: 25
  resampledPixelSpacing:
  interpolator: sitkBSpline
  correctMask: true
  label: 1
  minimumROIDimensions: 1
  minimumROISize: 1

featureClass:
  firstorder:
  glcm:
"""
    (out / "pyradiomics_minimal.yaml").write_text(config)


# ═══════════════════════════════════════════════════════════════
# Feature map fixture (for tabular column mapping)
# ═══════════════════════════════════════════════════════════════

def generate_feature_map() -> None:
    """Small feature map CSV matching the FDB fixture."""
    out = DATA_DIR / "fdb_ldb"
    out.mkdir(parents=True, exist_ok=True)

    lines = ["f_col;original_feature"]
    for i in range(8):
        lines.append(f"f{i};Modality::Group::feat{i}")
    (out / "feature_map.csv").write_text("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print(f"Generating fixtures in {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("  [1/9] Tabular binary...")
    generate_tabular_binary()

    print("  [2/9] Tabular multiclass...")
    generate_tabular_multiclass()

    print("  [3/9] Tabular regression...")
    generate_tabular_regression()

    print("  [4/9] NIfTI with masks...")
    generate_nifti_masked()

    print("  [5/9] NIfTI without masks...")
    generate_nifti_nomask()

    print("  [6/9] NIfTI multi-label masks...")
    generate_nifti_multilabel()

    print("  [7/9] FDB/LDB legacy format...")
    generate_fdb_ldb()
    generate_feature_map()

    print("  [8/9] TIFF synthetic...")
    generate_tiff_synthetic()

    print("  [9/9] PyRadiomics config...")
    generate_pyradiomics_config()

    # Summary
    total = sum(1 for _ in DATA_DIR.rglob("*") if _.is_file())
    print(f"\nDone. {total} fixture files in {DATA_DIR}")


if __name__ == "__main__":
    main()
