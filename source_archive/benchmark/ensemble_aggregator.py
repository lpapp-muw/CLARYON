#!/usr/bin/env python3
"""
ensemble_aggregator.py — Aggregate ensemble predictions from K DEBI-NN members.

Loads K Predictions.csv files for the same fold, verifies Key alignment,
averages softmax probabilities P0..PC-1, takes argmax for ensemble prediction,
and computes metrics.

Also extracts single-model (M0) results for K=1 comparison.

Output:
    results_dir/debinn/
    ├── single/seed_{s}/fold_{f}/{dataset}/Predictions.csv   (M0 only)
    └── ensemble/seed_{s}/fold_{f}/{dataset}/Predictions.csv  (averaged K members)
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from config import (
    RUNS_DIR, RESULTS_DIR, CSV_SEP, FLOAT_FMT,
    CV_SEEDS, N_FOLDS, DATASETS, LARGE_DATASET_THRESHOLD,
    DATASET_NAMES, ENSEMBLE_K, ALL_METRICS,
)


def load_predictions(pred_path):
    """Load a Predictions.csv file.

    Returns:
        keys (np.array), actuals (np.array[int]), prob_cols (list[str]), probs (np.array)
    """
    df = pd.read_csv(pred_path, sep=CSV_SEP)
    keys = df["Key"].values
    actuals = df["Actual"].values.astype(int)
    prob_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
    prob_cols = sorted(prob_cols, key=lambda c: int(c[1:]))
    probs = df[prob_cols].values.astype(np.float64)
    return keys, actuals, prob_cols, probs


def find_member_predictions(project_dir, dataset_name, fold_idx, k):
    """Locate Predictions.csv for each ensemble member.

    Returns:
        list of paths (length k), or raises FileNotFoundError.
    """
    finished_dir = os.path.join(project_dir, "Executions-Finished")
    paths = []
    for member_idx in range(k):
        exec_name = f"{dataset_name}-M{member_idx}"
        pred_path = os.path.join(
            finished_dir, exec_name, "Log",
            f"Fold-{fold_idx + 1}", "Predictions.csv")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing: {pred_path}")
        paths.append(pred_path)
    return paths


def aggregate_ensemble(pred_paths):
    """Average softmax across K member Predictions.csv files.

    Returns:
        keys, actuals, prob_cols, averaged_probs, ensemble_preds
    """
    all_probs = []
    ref_keys = None
    ref_actuals = None
    ref_prob_cols = None

    for path in pred_paths:
        keys, actuals, prob_cols, probs = load_predictions(path)

        if ref_keys is None:
            ref_keys = keys
            ref_actuals = actuals
            ref_prob_cols = prob_cols
        else:
            # Verify alignment.
            assert np.array_equal(ref_keys, keys), \
                f"Key mismatch between {pred_paths[0]} and {path}"
            assert np.array_equal(ref_actuals, actuals), \
                f"Actual label mismatch between {pred_paths[0]} and {path}"

        all_probs.append(probs)

    # Average softmax probabilities.
    stacked = np.stack(all_probs, axis=0)  # (K, N, C)
    avg_probs = stacked.mean(axis=0)       # (N, C)

    # Argmax for ensemble prediction.
    ensemble_preds = np.argmax(avg_probs, axis=1)

    return ref_keys, ref_actuals, ref_prob_cols, avg_probs, ensemble_preds


def compute_metrics(y_true, y_pred, probs):
    """Compute standard metrics from predictions and probabilities."""
    n_classes = probs.shape[1]
    bacc = balanced_accuracy_score(y_true, y_pred)
    acc = np.mean(y_pred == y_true)

    probs_clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)
    probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
    ll = log_loss(y_true, probs_clipped, labels=list(range(n_classes)))

    return {"BACC": bacc, "ACC": acc, "EntropyLoss": ll}


def save_predictions_csv(keys, actuals, preds, probs, prob_cols, out_path):
    """Save predictions in DEBI-NN format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    for i in range(len(keys)):
        row = {"Key": keys[i], "Actual": actuals[i], "Predicted": preds[i]}
        for j, col in enumerate(prob_cols):
            row[col] = f"{probs[i, j]:.8f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep=CSV_SEP, index=False)


def aggregate_dataset_seed(dataset_name, seed, k=None):
    """Aggregate ensemble for all folds of a dataset/seed pair.

    Returns:
        list of dicts with fold_idx, single_metrics, ensemble_metrics.
    """
    if k is None:
        k = ENSEMBLE_K

    project_dir = os.path.join(RUNS_DIR, "projects",
                               f"project_{dataset_name}_seed{seed}")
    is_large = DATASETS[dataset_name]["n"] > LARGE_DATASET_THRESHOLD
    n_folds = 1 if is_large else N_FOLDS

    fold_results = []

    for fold_idx in range(n_folds):
        pred_paths = find_member_predictions(
            project_dir, dataset_name, fold_idx, k)

        # Single model (M0).
        keys_s, actuals_s, prob_cols_s, probs_s = load_predictions(pred_paths[0])
        preds_s = np.argmax(probs_s, axis=1)
        single_metrics = compute_metrics(actuals_s, preds_s, probs_s)

        single_out = os.path.join(
            RESULTS_DIR, "debinn", "single",
            f"seed_{seed}", f"fold_{fold_idx}", dataset_name,
            "Predictions.csv")
        save_predictions_csv(
            keys_s, actuals_s, preds_s, probs_s, prob_cols_s, single_out)

        # Ensemble (all K members).
        keys_e, actuals_e, prob_cols_e, avg_probs, ens_preds = \
            aggregate_ensemble(pred_paths)
        ensemble_metrics = compute_metrics(actuals_e, ens_preds, avg_probs)

        ensemble_out = os.path.join(
            RESULTS_DIR, "debinn", "ensemble",
            f"seed_{seed}", f"fold_{fold_idx}", dataset_name,
            "Predictions.csv")
        save_predictions_csv(
            keys_e, actuals_e, ens_preds, avg_probs, prob_cols_e, ensemble_out)

        fold_results.append({
            "fold": fold_idx,
            "single": single_metrics,
            "ensemble": ensemble_metrics,
        })

    return fold_results


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate DEBI-NN ensemble predictions")
    parser.add_argument("--dataset", default=None, help="Single dataset")
    parser.add_argument("--all", action="store_true", help="All datasets")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--k", type=int, default=ENSEMBLE_K)
    args = parser.parse_args()

    if not args.dataset and not args.all:
        print("ERROR: Specify --dataset or --all")
        return

    datasets = [args.dataset] if args.dataset else DATASET_NAMES
    seeds = args.seeds if args.seeds else CV_SEEDS

    print(f"Ensemble Aggregator | K={args.k}")
    print(f"{'=' * 60}")

    for ds in datasets:
        if ds not in DATASETS:
            continue

        for seed in seeds:
            print(f"  [{ds}] seed={seed} ", end="", flush=True)
            try:
                fold_results = aggregate_dataset_seed(ds, seed, args.k)
                single_bacc = np.mean([r["single"]["BACC"] for r in fold_results])
                ens_bacc = np.mean([r["ensemble"]["BACC"] for r in fold_results])
                delta = ens_bacc - single_bacc
                sign = "+" if delta >= 0 else ""
                print(f"single={single_bacc:.3f}  ensemble={ens_bacc:.3f}  "
                      f"({sign}{delta:.3f})")
            except Exception as e:
                print(f"FAILED ({e})")

    print("\nDone.")


if __name__ == "__main__":
    main()
