#!/usr/bin/env python3
"""
fold_generator.py — Generate stratified k-fold CV splits for the benchmark.

Standard datasets (N <= 10K): StratifiedKFold, 5 folds × 3 seeds.
Large datasets   (N > 10K):   Fixed stratified 60/20/20 split × 3 seeds.

Reads FDB.csv + LDB.csv from the preprocessed directory.
Outputs TrainF/TrainL/TestF/TestL per fold into the benchmark_runs tree.

Output structure:
    benchmark_runs/
    ├── seed_42/
    │   ├── fold_0/
    │   │   ├── iris/
    │   │   │   ├── TrainF.csv
    │   │   │   ├── TrainL.csv
    │   │   │   ├── TestF.csv
    │   │   │   └── TestL.csv
    │   │   ├── diabetes/
    │   │   ...
    │   ├── fold_1/
    │   ...
    ├── seed_123/
    ...

For large datasets, fold_0 is the single split (no fold_1..fold_4).

Reuses FDB/LDB format conventions from E5 (semicolon separator, Key column).
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from config import (
    PREPROCESSED_DIR, RUNS_DIR, CSV_SEP, FLOAT_FMT,
    CV_SEEDS, N_FOLDS, DATASETS, LARGE_DATASET_THRESHOLD,
    LARGE_SPLIT_RATIOS, DATASET_NAMES,
)


def load_dataset(dataset_name):
    """Load FDB.csv and LDB.csv for a dataset. Returns (fdb_df, ldb_df)."""
    ds_dir = os.path.join(PREPROCESSED_DIR, dataset_name)
    fdb_path = os.path.join(ds_dir, "FDB.csv")
    ldb_path = os.path.join(ds_dir, "LDB.csv")

    if not os.path.exists(fdb_path) or not os.path.exists(ldb_path):
        raise FileNotFoundError(f"FDB.csv or LDB.csv not found in {ds_dir}")

    fdb = pd.read_csv(fdb_path, sep=CSV_SEP)
    ldb = pd.read_csv(ldb_path, sep=CSV_SEP)

    assert len(fdb) == len(ldb), f"FDB/LDB row count mismatch: {len(fdb)} vs {len(ldb)}"
    assert list(fdb["Key"]) == list(ldb["Key"]), "FDB/LDB Key column mismatch"

    return fdb, ldb


def save_split(fdb, ldb, train_idx, test_idx, out_dir):
    """Save a single train/test split to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    train_fdb = fdb.iloc[train_idx].reset_index(drop=True)
    train_ldb = ldb.iloc[train_idx].reset_index(drop=True)
    test_fdb = fdb.iloc[test_idx].reset_index(drop=True)
    test_ldb = ldb.iloc[test_idx].reset_index(drop=True)

    train_fdb.to_csv(os.path.join(out_dir, "TrainF.csv"),
                     sep=CSV_SEP, index=False, float_format=FLOAT_FMT)
    train_ldb.to_csv(os.path.join(out_dir, "TrainL.csv"),
                     sep=CSV_SEP, index=False)
    test_fdb.to_csv(os.path.join(out_dir, "TestF.csv"),
                    sep=CSV_SEP, index=False, float_format=FLOAT_FMT)
    test_ldb.to_csv(os.path.join(out_dir, "TestL.csv"),
                    sep=CSV_SEP, index=False)

    return len(train_fdb), len(test_fdb)


def generate_kfold_splits(fdb, ldb, dataset_name, seed):
    """Generate N_FOLDS stratified folds for a standard dataset."""
    y = ldb["Label"].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    fold_info = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(fdb, y)):
        out_dir = os.path.join(RUNS_DIR, f"seed_{seed}", f"fold_{fold_idx}", dataset_name)
        n_train, n_test = save_split(fdb, ldb, train_idx, test_idx, out_dir)
        fold_info.append({"fold": fold_idx, "n_train": n_train, "n_test": n_test})

    return fold_info


def generate_large_split(fdb, ldb, dataset_name, seed):
    """Generate a single fixed 60/20/20 split for large datasets.

    The 20% validation portion is NOT written as separate files — DEBI-NN's
    internal auto-split (Validation/AutoSplit=TRUE) handles train→train+val.
    We write 80% as Train (60% train + 20% val after auto-split) and 20% as Test.
    """
    y = ldb["Label"].values
    test_ratio = LARGE_SPLIT_RATIOS[2]  # 0.2

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_idx, test_idx = next(sss.split(fdb, y))

    out_dir = os.path.join(RUNS_DIR, f"seed_{seed}", "fold_0", dataset_name)
    n_train, n_test = save_split(fdb, ldb, train_idx, test_idx, out_dir)

    return [{"fold": 0, "n_train": n_train, "n_test": n_test}]


def generate_splits_for_dataset(dataset_name, seeds=None):
    """Generate all splits for a dataset across all seeds."""
    if seeds is None:
        seeds = CV_SEEDS

    fdb, ldb = load_dataset(dataset_name)
    is_large = DATASETS[dataset_name]["n"] > LARGE_DATASET_THRESHOLD

    results = {}
    for seed in seeds:
        if is_large:
            fold_info = generate_large_split(fdb, ldb, dataset_name, seed)
        else:
            fold_info = generate_kfold_splits(fdb, ldb, dataset_name, seed)
        results[seed] = fold_info

    return results, is_large


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified CV folds for DEBI-NN benchmark")
    parser.add_argument("--dataset", default=None,
                        help="Process a single dataset by name")
    parser.add_argument("--all", action="store_true",
                        help="Process all 28 benchmark datasets")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help=f"Seeds to use (default: {CV_SEEDS})")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        print("ERROR: Specify --dataset <name> or --all")
        return

    datasets = [args.dataset] if args.dataset else DATASET_NAMES
    seeds = args.seeds if args.seeds else CV_SEEDS

    print(f"Fold Generator | seeds={seeds} | n_folds={N_FOLDS}")
    print(f"  Large dataset threshold: N > {LARGE_DATASET_THRESHOLD}")
    print(f"  Output: {RUNS_DIR}")
    print(f"{'=' * 60}")

    summary = {}
    for ds in datasets:
        if ds not in DATASETS:
            print(f"  [{ds}] SKIPPED — not in dataset registry")
            continue

        print(f"  [{ds}] ", end="", flush=True)
        try:
            results, is_large = generate_splits_for_dataset(ds, seeds)
            tag = "LARGE (fixed split)" if is_large else f"{N_FOLDS}-fold CV"
            total_folds = sum(len(v) for v in results.values())
            print(f"OK  ({tag}, {total_folds} total splits)")
            summary[ds] = {"type": tag, "splits_per_seed": results}
        except Exception as e:
            print(f"FAILED  ({e})")

    # Save generation manifest.
    manifest_path = os.path.join(RUNS_DIR, "fold_manifest.json")
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({
            "seeds": seeds,
            "n_folds": N_FOLDS,
            "large_threshold": LARGE_DATASET_THRESHOLD,
            "datasets": {ds: {"type": info["type"]} for ds, info in summary.items()},
        }, f, indent=2)

    print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
