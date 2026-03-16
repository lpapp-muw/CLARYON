#!/usr/bin/env python3
"""
results_collector.py — Collect all results into a unified results table.

Walks DEBI-NN (single + ensemble) and competitor Predictions.csv files,
computes metrics per dataset/method/seed/fold, and writes results_table.csv.

Output columns:
    dataset, method, seed, fold, ACC, BACC, EntropyLoss, MacroF1, WeightedF1
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, log_loss,
    matthews_corrcoef, f1_score,
)

from config import (
    RUNS_DIR, RESULTS_DIR, CSV_SEP,
    CV_SEEDS, N_FOLDS, DATASETS, LARGE_DATASET_THRESHOLD,
    DATASET_NAMES, COMPETITORS, DEBINN_METHOD_NAMES,
)


def load_predictions(pred_path):
    """Load Predictions.csv → (y_true, y_pred, probs)."""
    df = pd.read_csv(pred_path, sep=CSV_SEP)
    y_true = df["Actual"].values.astype(int)
    y_pred = df["Predicted"].values.astype(int)
    prob_cols = sorted(
        [c for c in df.columns if c.startswith("P") and c[1:].isdigit()],
        key=lambda c: int(c[1:]))
    probs = df[prob_cols].values.astype(np.float64)
    return y_true, y_pred, probs


def compute_full_metrics(y_true, y_pred, probs):
    """Compute all benchmark metrics."""
    n_classes = probs.shape[1]

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    probs_c = np.clip(probs, 1e-15, 1.0 - 1e-15)
    probs_c = probs_c / probs_c.sum(axis=1, keepdims=True)
    entropy_loss = log_loss(y_true, probs_c, labels=list(range(n_classes)))

    mcc = matthews_corrcoef(y_true, y_pred)

    avg = "macro" if n_classes > 2 else "binary"
    pos = 1 if n_classes == 2 else None
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "ACC": acc,
        "BACC": bacc,
        "EntropyLoss": entropy_loss,
        "MCC": mcc,
        "MacroF1": macro_f1,
        "WeightedF1": weighted_f1,
    }


def get_fold_count(dataset_name):
    """Return number of folds for a dataset."""
    if DATASETS[dataset_name]["n"] > LARGE_DATASET_THRESHOLD:
        return 1
    return N_FOLDS


def collect_debinn_results(datasets, seeds):
    """Collect DEBI-NN single and ensemble results."""
    rows = []

    for ds in datasets:
        n_folds = get_fold_count(ds)
        for seed in seeds:
            for fold_idx in range(n_folds):
                # Single model.
                single_path = os.path.join(
                    RESULTS_DIR, "debinn", "single",
                    f"seed_{seed}", f"fold_{fold_idx}", ds, "Predictions.csv")
                if os.path.exists(single_path):
                    y_true, y_pred, probs = load_predictions(single_path)
                    metrics = compute_full_metrics(y_true, y_pred, probs)
                    rows.append({
                        "dataset": ds, "method": "DEBINN-single",
                        "seed": seed, "fold": fold_idx, **metrics})

                # Ensemble.
                ens_path = os.path.join(
                    RESULTS_DIR, "debinn", "ensemble",
                    f"seed_{seed}", f"fold_{fold_idx}", ds, "Predictions.csv")
                if os.path.exists(ens_path):
                    y_true, y_pred, probs = load_predictions(ens_path)
                    metrics = compute_full_metrics(y_true, y_pred, probs)
                    rows.append({
                        "dataset": ds, "method": "DEBINN-ensemble",
                        "seed": seed, "fold": fold_idx, **metrics})

    return rows


def collect_competitor_results(datasets, seeds):
    """Collect competitor results."""
    rows = []

    for ds in datasets:
        n_folds = get_fold_count(ds)
        for seed in seeds:
            for fold_idx in range(n_folds):
                for method in COMPETITORS:
                    pred_path = os.path.join(
                        RESULTS_DIR, "competitors", method,
                        f"seed_{seed}", f"fold_{fold_idx}", ds,
                        "Predictions.csv")
                    if os.path.exists(pred_path):
                        try:
                            y_true, y_pred, probs = load_predictions(pred_path)
                            metrics = compute_full_metrics(y_true, y_pred, probs)
                            rows.append({
                                "dataset": ds, "method": method,
                                "seed": seed, "fold": fold_idx, **metrics})
                        except Exception as e:
                            print(f"  WARNING: {method}/{ds}/seed{seed}/fold{fold_idx}: {e}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Collect all benchmark results into a unified table")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: results_dir/results_table.csv)")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        print("ERROR: Specify --dataset or --all")
        return

    datasets = [args.dataset] if args.dataset else DATASET_NAMES
    seeds = args.seeds if args.seeds else CV_SEEDS
    output_path = args.output or os.path.join(RESULTS_DIR, "results_table.csv")

    print("Collecting results...")

    debinn_rows = collect_debinn_results(datasets, seeds)
    print(f"  DEBI-NN: {len(debinn_rows)} entries")

    comp_rows = collect_competitor_results(datasets, seeds)
    print(f"  Competitors: {len(comp_rows)} entries")

    all_rows = debinn_rows + comp_rows
    if not all_rows:
        print("  No results found.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["dataset", "method", "seed", "fold"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults table: {output_path}")
    print(f"  {len(df)} rows, {df['method'].nunique()} methods, "
          f"{df['dataset'].nunique()} datasets")

    # Quick summary: mean BACC per method.
    print(f"\n{'Method':<20} {'Mean BACC':>10} {'Datasets':>10}")
    print("-" * 42)
    summary = df.groupby("method")["BACC"].agg(["mean", "count"])
    summary = summary.sort_values("mean", ascending=False)
    for method, row in summary.iterrows():
        print(f"  {method:<18} {row['mean']:>10.4f} {int(row['count']):>10}")


if __name__ == "__main__":
    main()
