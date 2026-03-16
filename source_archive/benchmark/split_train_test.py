#!/usr/bin/env python3
"""
split_train_test.py — Generate Train/Test split files for DEBI-NN

Takes preprocessed FDB.csv + LDB.csv and produces:
  - TrainF.csv, TrainL.csv  (80% stratified)
  - TestF.csv,  TestL.csv   (20% stratified)

DEBI-NN's internal auto-split (Validation/AutoSplit=TRUE) will further
split TrainFDB/TrainLDB into train+validate during training.

Usage:
    python split_train_test.py --dataset iris
    python split_train_test.py --dataset iris --seed 42 --test-ratio 0.2
    python split_train_test.py --all --seed 42

The output goes into the same dataset folder (benchmark_preprocessed/<dataset>/).
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def split_dataset(dataset_dir, seed=42, test_ratio=0.2):
    """Split a single dataset's FDB/LDB into Train/Test."""

    fdb_path = os.path.join(dataset_dir, "FDB.csv")
    ldb_path = os.path.join(dataset_dir, "LDB.csv")

    if not os.path.exists(fdb_path) or not os.path.exists(ldb_path):
        raise FileNotFoundError(f"FDB.csv or LDB.csv not found in {dataset_dir}")

    fdb = pd.read_csv(fdb_path, sep=";")
    ldb = pd.read_csv(ldb_path, sep=";")

    assert len(fdb) == len(ldb), f"FDB/LDB row count mismatch: {len(fdb)} vs {len(ldb)}"
    assert list(fdb["Key"]) == list(ldb["Key"]), "FDB/LDB key mismatch"

    y = ldb["Label"].values

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_idx, test_idx = next(sss.split(fdb, y))

    # Split
    train_fdb = fdb.iloc[train_idx].reset_index(drop=True)
    train_ldb = ldb.iloc[train_idx].reset_index(drop=True)
    test_fdb = fdb.iloc[test_idx].reset_index(drop=True)
    test_ldb = ldb.iloc[test_idx].reset_index(drop=True)

    # Save
    train_fdb.to_csv(os.path.join(dataset_dir, "TrainF.csv"), sep=";", index=False, float_format="%.8f")
    train_ldb.to_csv(os.path.join(dataset_dir, "TrainL.csv"), sep=";", index=False)
    test_fdb.to_csv(os.path.join(dataset_dir, "TestF.csv"), sep=";", index=False, float_format="%.8f")
    test_ldb.to_csv(os.path.join(dataset_dir, "TestL.csv"), sep=";", index=False)

    # Class distribution check
    train_dist = pd.Series(train_ldb["Label"]).value_counts().sort_index().to_dict()
    test_dist = pd.Series(test_ldb["Label"]).value_counts().sort_index().to_dict()

    split_info = {
        "seed": seed,
        "test_ratio": test_ratio,
        "n_train": len(train_fdb),
        "n_test": len(test_fdb),
        "train_class_distribution": train_dist,
        "test_class_distribution": test_dist,
    }

    with open(os.path.join(dataset_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2, default=str)

    return split_info


def main():
    parser = argparse.ArgumentParser(description="Split preprocessed datasets into Train/Test for DEBI-NN")
    parser.add_argument("--base-dir", default="benchmark_preprocessed",
                        help="Base directory containing preprocessed datasets")
    parser.add_argument("--dataset", default=None,
                        help="Process a single dataset by name")
    parser.add_argument("--all", action="store_true",
                        help="Process all datasets in base-dir")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split (default: 42)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Test set ratio (default: 0.2)")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        print("ERROR: Specify --dataset <name> or --all")
        return

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted([
            d for d in os.listdir(args.base_dir)
            if os.path.isdir(os.path.join(args.base_dir, d))
        ])

    print(f"Train/Test Splitter | seed={args.seed} | test_ratio={args.test_ratio}")
    print(f"{'=' * 60}")

    for ds in datasets:
        ds_dir = os.path.join(args.base_dir, ds)
        print(f"  [{ds}] ", end="", flush=True)
        try:
            info = split_dataset(ds_dir, seed=args.seed, test_ratio=args.test_ratio)
            print(f"OK  (train={info['n_train']}, test={info['n_test']})")
        except Exception as e:
            print(f"FAILED  ({e})")

    print(f"\nDone. Output files per dataset: TrainF.csv, TrainL.csv, TestF.csv, TestL.csv")


if __name__ == "__main__":
    main()
