#!/usr/bin/env python3
"""Prepare demo experiments for CLARYON.

Creates:
  1. Iris dataset in CLARYON CSV format (semicolon-separated)
  2. Synthetic NIfTI volumes with masks (if not already present)
  3. YAML configs for both experiments

Run from project root:
    cd ~/claryon
    source .venv/bin/activate
    python scripts/setup_experiments.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "demo_data"


def prepare_iris() -> Path:
    """Download iris and write as semicolon-separated CSV."""
    out = DATA_DIR / "iris"
    out.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=[f"f{i}" for i in range(4)])
    df["label"] = iris.target  # 0, 1, 2 (multiclass)

    # Add Key column
    df.insert(0, "Key", [f"S{i:04d}" for i in range(len(df))])

    path = out / "iris.csv"
    df.to_csv(path, sep=";", index=False)
    print(f"  Iris: {path} ({len(df)} samples, {iris.data.shape[1]} features, 3 classes)")

    # Also make a binary version (setosa vs non-setosa)
    df_bin = df.copy()
    df_bin["label"] = (df_bin["label"] > 0).astype(int)
    bin_path = out / "iris_binary.csv"
    df_bin.to_csv(bin_path, sep=";", index=False)
    print(f"  Iris binary: {bin_path} ({len(df_bin)} samples, 2 classes)")

    return out


def prepare_nifti() -> Path:
    """Generate synthetic NIfTI volumes if not present."""
    import nibabel as nib

    out = DATA_DIR / "nifti_demo"
    if (out / "Train").exists() and len(list((out / "Train").glob("*.nii.gz"))) > 0:
        print(f"  NIfTI: already exists at {out}")
        return out

    shape = (16, 16, 16)
    rng = np.random.default_rng(42)

    for split, n in [("Train", 20), ("Test", 10)]:
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            label = i % 2
            pet = rng.normal(loc=0.0, scale=0.3, size=shape).astype(np.float32)
            mask = np.zeros(shape, dtype=np.float32)
            mask[4:12, 4:12, 4:12] = 1.0

            if label == 1:
                pet += mask * 2.0

            case = f"case{i:03d}_{label}"
            nib.save(
                nib.Nifti1Image(pet, np.eye(4, dtype=np.float32)),
                str(d / f"{case}_PET.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(mask, np.eye(4, dtype=np.float32)),
                str(d / f"{case}_mask.nii.gz"),
            )

    print(f"  NIfTI: {out} (20 train + 10 test, shape {shape})")
    return out


def write_configs(iris_dir: Path, nifti_dir: Path) -> None:
    """Write YAML config files."""
    configs_dir = ROOT / "configs"
    configs_dir.mkdir(exist_ok=True)

    # ── Iris multiclass: XGBoost + LightGBM + CatBoost ──
    iris_classical = f"""\
experiment:
  name: iris_classical
  seed: 42
  results_dir: Results/iris_classical

data:
  tabular:
    path: {iris_dir / 'iris.csv'}
    label_col: label
    id_col: Key
    sep: ";"

cv:
  strategy: kfold
  n_folds: 5
  seeds: [42]

models:
  - name: xgboost
    type: tabular
    params:
      n_estimators: 100
  - name: lightgbm
    type: tabular
  - name: catboost
    type: tabular
    params:
      iterations: 100
      verbose: 0

evaluation:
  metrics: [bacc, auc, sensitivity, specificity]

reporting:
  markdown: true
"""
    p = configs_dir / "iris_classical.yaml"
    p.write_text(iris_classical)
    print(f"  Config: {p}")

    # ── Iris binary: quantum kernel SVM + qCNN ──
    iris_quantum = f"""\
experiment:
  name: iris_quantum
  seed: 42
  results_dir: Results/iris_quantum

data:
  tabular:
    path: {iris_dir / 'iris_binary.csv'}
    label_col: label
    id_col: Key
    sep: ";"

cv:
  strategy: holdout
  test_size: 0.25
  seeds: [42]

models:
  - name: kernel_svm
    type: tabular_quantum
    params:
      n_qubits: 3
  - name: qcnn_muw
    type: tabular_quantum
    params:
      n_qubits: 3
      epochs: 10
      lr: 0.02
  - name: qcnn_alt
    type: tabular_quantum
    params:
      n_qubits: 3
      epochs: 10
      lr: 0.02

evaluation:
  metrics: [bacc, auc, sensitivity, specificity]

reporting:
  markdown: true
"""
    p = configs_dir / "iris_quantum.yaml"
    p.write_text(iris_quantum)
    print(f"  Config: {p}")

    # ── NIfTI: 3D CNN ──
    nifti_cnn = f"""\
experiment:
  name: nifti_cnn
  seed: 42
  results_dir: Results/nifti_cnn

data:
  imaging:
    path: {nifti_dir}
    format: nifti
    mask_pattern: "*mask*"

cv:
  strategy: kfold
  n_folds: 3
  seeds: [42]

models:
  - name: cnn_3d
    type: imaging
    params:
      epochs: 10
      batch_size: 4

evaluation:
  metrics: [bacc, auc]

reporting:
  markdown: true
"""
    p = configs_dir / "nifti_cnn.yaml"
    p.write_text(nifti_cnn)
    print(f"  Config: {p}")


def main() -> None:
    print("Setting up CLARYON demo experiments...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Preparing iris dataset...")
    iris_dir = prepare_iris()

    print("\n[2/3] Preparing NIfTI volumes...")
    nifti_dir = prepare_nifti()

    print("\n[3/3] Writing configs...")
    write_configs(iris_dir, nifti_dir)

    print("\n" + "=" * 60)
    print("Ready. Run experiments with:")
    print()
    print("  # Classical models on iris (fast, ~1 min)")
    print("  python -m claryon run -c configs/iris_classical.yaml -vv")
    print()
    print("  # Quantum models on iris binary (slow, ~10-30 min)")
    print("  python -m claryon run -c configs/iris_quantum.yaml -vv")
    print()
    print("  # 3D CNN on NIfTI (medium, ~5 min)")
    print("  python -m claryon run -c configs/nifti_cnn.yaml -vv")
    print()
    print("Results appear in Results/<experiment_name>/")
    print("=" * 60)


if __name__ == "__main__":
    main()
