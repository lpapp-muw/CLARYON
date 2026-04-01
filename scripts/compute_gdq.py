#!/usr/bin/env python3
"""Compute Geometric Difference (GDQ) score across all benchmark splits.

Builds the quantum kernel matrix (amplitude fidelity kernel) for each fold,
runs the Huang et al. 2021 analysis against linear/RBF/polynomial classical
kernels, and reports per-fold and aggregated results.

Note: GDQ evaluates amplitude-encoded fidelity kernels only. It does not
apply to projected quantum kernels (angle_pqk_svm, projected_kernel_svm)
or training-based models (qnn, qcnn_muw, qcnn_alt).

Usage:
    cd ~/claryon && source .venv/bin/activate
    python scripts/compute_gdq.py

Results written to Results/eanm_abstract/<dataset>/gdq_summary.csv
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claryon.encoding.amplitude import amplitude_encode_matrix
from claryon.evaluation.geometric_difference import (
    geometric_difference_score,
    model_complexity,
    effective_dimension,
    generate_gdq_report,
)
from claryon.preprocessing.feature_selection import mrmr_select
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Datasets to analyze — use q8 results (same preprocessing)
DATASETS = {
    "wisconsin_q8": {
        "data": "datasets/wisconsin-breast-cancer/train.csv",
        "results": "Results/eanm_abstract/wisconsin_q8",
    },
    "hcc_q8": {
        "data": "datasets/hcc-survival/train.csv",
        "results": "Results/eanm_abstract/hcc_q8",
    },
    "psma11_q8": {
        "data": "datasets/psma11/train.csv",
        "results": "Results/eanm_abstract/psma11_q8",
    },
}


def build_quantum_kernel(X_encoded: np.ndarray) -> np.ndarray:
    """Build amplitude fidelity kernel: K[i,j] = |<x_i|x_j>|^2.

    This is the same kernel used by kernel_svm and quantum_gp.
    Since X_encoded is already L2-normalized, K[i,j] = (x_i · x_j)^2.
    """
    inner = X_encoded @ X_encoded.T
    return inner ** 2


def analyze_dataset(name: str, data_path: str, results_dir: str) -> pd.DataFrame:
    """Run GDQ analysis for all folds of a dataset."""
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning("Results not found: %s — skipping", results_dir)
        return pd.DataFrame()

    # Load data
    df = pd.read_csv(data_path, sep=";")
    label_col = "label"
    feature_cols = [c for c in df.columns if c not in (label_col, "Key")]
    X_all = df[feature_cols].values.astype(np.float64)
    y_all = df[label_col].values.astype(int)
    feature_names = feature_cols

    # Find all preprocessing states to get the exact splits/features used
    kernel_svm_dir = results_path / "kernel_svm"
    if not kernel_svm_dir.exists():
        logger.warning("No kernel_svm results in %s — skipping", results_dir)
        return pd.DataFrame()

    rows = []

    for seed_dir in sorted(kernel_svm_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        seed = int(seed_dir.name.split("_")[1])

        for fold_dir in sorted(seed_dir.iterdir()):
            if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
                continue
            fold = int(fold_dir.name.split("_")[1])

            # Load preprocessing state to get selected features
            state_path = fold_dir / "preprocessing_state.json"
            if not state_path.exists():
                logger.warning("Missing state: %s", state_path)
                continue

            with open(state_path) as f:
                state = json.load(f)

            selected_idx = state["selected_features"]

            # Load predictions to get train/test split indices
            pred_path = fold_dir / "Predictions.csv"
            if not pred_path.exists():
                continue

            pred_df = pd.read_csv(pred_path, sep=";")
            test_keys = set(pred_df["Key"].values)

            # Reconstruct train/test split
            if "Key" in df.columns:
                all_keys = df["Key"].values
                test_mask = np.isin(all_keys, list(test_keys))
            else:
                all_keys = [f"S{i:04d}" for i in range(len(df))]
                test_mask = np.isin(all_keys, list(test_keys))

            train_idx = np.where(~test_mask)[0]

            X_train = X_all[train_idx][:, selected_idx]
            y_train = y_all[train_idx]

            # Amplitude encode (same as pipeline)
            X_encoded, enc_info = amplitude_encode_matrix(X_train)

            # Build quantum kernel
            K_Q = build_quantum_kernel(X_encoded)

            # Classical kernels
            n_feat = X_encoded.shape[1]
            K_linear = linear_kernel(X_encoded)
            K_rbf = rbf_kernel(X_encoded, gamma=1.0 / n_feat)
            K_poly = polynomial_kernel(X_encoded, degree=3)

            # GDQ scores
            y_float = y_train.astype(np.float64)
            # Encode labels as +1/-1 for complexity
            y_signed = np.where(y_float == 1, 1.0, -1.0)

            try:
                g_linear = geometric_difference_score(K_Q, K_linear)
            except Exception:
                g_linear = float("nan")
            try:
                g_rbf = geometric_difference_score(K_Q, K_rbf)
            except Exception:
                g_rbf = float("nan")
            try:
                g_poly = geometric_difference_score(K_Q, K_poly)
            except Exception:
                g_poly = float("nan")

            # Model complexities
            try:
                s_Q = model_complexity(K_Q, y_signed)
            except Exception:
                s_Q = float("nan")
            try:
                s_linear = model_complexity(K_linear, y_signed)
            except Exception:
                s_linear = float("nan")
            try:
                s_rbf = model_complexity(K_rbf, y_signed)
            except Exception:
                s_rbf = float("nan")

            # Effective dimension
            d = effective_dimension(K_Q)

            # Decision
            max_g = max(g for g in [g_linear, g_rbf, g_poly] if g == g)  # noqa
            if max_g < 1.1:
                rec = "classical_sufficient"
            elif s_Q < min(s_linear, s_rbf) * 0.5:
                rec = "quantum_advantage_likely"
            else:
                rec = "inconclusive"

            rows.append({
                "dataset": name,
                "seed": seed,
                "fold": fold,
                "n_train": len(train_idx),
                "n_features": len(selected_idx),
                "n_qubits": enc_info.n_qubits,
                "g_linear": round(g_linear, 4),
                "g_rbf": round(g_rbf, 4),
                "g_poly": round(g_poly, 4),
                "g_max": round(max_g, 4),
                "s_Q": round(s_Q, 4),
                "s_linear": round(s_linear, 4),
                "s_rbf": round(s_rbf, 4),
                "d_eff": d,
                "recommendation": rec,
            })

            logger.info(
                "  seed=%d fold=%d g_max=%.3f s_Q=%.3f rec=%s",
                seed, fold, max_g, s_Q, rec,
            )

    return pd.DataFrame(rows)


def main() -> None:
    logger.info("=" * 60)
    logger.info("CLARYON — Geometric Difference (GDQ) Analysis")
    logger.info("=" * 60)

    all_results = []

    for name, cfg in DATASETS.items():
        results_path = Path(cfg["results"])
        if not results_path.exists():
            logger.info("\n%s: results not found, skipping", name)
            continue

        logger.info("\n=== %s ===", name)
        df = analyze_dataset(name, cfg["data"], cfg["results"])
        if df.empty:
            continue

        all_results.append(df)

        # Per-dataset summary
        logger.info("\n  Summary for %s:", name)
        logger.info("    g_max:  %.3f ± %.3f", df["g_max"].mean(), df["g_max"].std())
        logger.info("    s_Q:    %.3f ± %.3f", df["s_Q"].mean(), df["s_Q"].std())
        logger.info("    d_eff:  %.1f ± %.1f", df["d_eff"].mean(), df["d_eff"].std())

        rec_counts = df["recommendation"].value_counts()
        for rec, count in rec_counts.items():
            logger.info("    %s: %d/%d folds", rec, count, len(df))

        # Save per-dataset
        out = results_path / "gdq_summary.csv"
        df.to_csv(out, sep=";", index=False)
        logger.info("    Saved to %s", out)

        # Generate plot for one representative fold
        try:
            from claryon.evaluation.geometric_difference import quantum_advantage_analysis
            # Use first fold data for the report
            row0 = df.iloc[0]
            analysis = {
                "g_CQ": {"linear": row0["g_linear"], "rbf": row0["g_rbf"], "polynomial": row0["g_poly"]},
                "s_C": {"linear": row0["s_linear"], "rbf": row0["s_rbf"]},
                "s_Q": row0["s_Q"],
                "d": row0["d_eff"],
                "K_Q_rank": row0["d_eff"],
                "recommendation": row0["recommendation"],
                "explanation": f"g_max={row0['g_max']:.2f}, s_Q={row0['s_Q']:.2f}",
            }
            generate_gdq_report(analysis, results_path)
        except Exception as e:
            logger.warning("    Plot generation failed: %s", e)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out = Path("Results/eanm_abstract/gdq_all_datasets.csv")
        combined.to_csv(out, sep=";", index=False)
        logger.info("\n" + "=" * 60)
        logger.info("Combined results saved to %s", out)

        # Overall summary
        logger.info("\n=== OVERALL SUMMARY ===")
        for ds_name in combined["dataset"].unique():
            ds = combined[combined["dataset"] == ds_name]
            logger.info(
                "  %s: g_max=%.3f±%.3f  rec=%s (%d folds)",
                ds_name,
                ds["g_max"].mean(),
                ds["g_max"].std(),
                ds["recommendation"].mode().iloc[0],
                len(ds),
            )
    else:
        logger.info("\nNo results to analyze.")


if __name__ == "__main__":
    main()
