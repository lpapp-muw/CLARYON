#!/usr/bin/env python3
"""
project_builder.py — Assemble BatchManager-compatible project folders.

For each (dataset, seed) pair, creates a project folder with:
    project_{dataset}_seed{s}/
    ├── executionSettings.csv   (K columns, from settings_generator)
    └── DataPool/
        └── {dataset}/
            └── Dataset/
                ├── Fold-1/
                │   ├── TrainF.csv  (from fold_generator output)
                │   ├── TrainL.csv
                │   ├── TestF.csv
                │   └── TestL.csv
                ├── Fold-2/
                ...

The C++ binary is invoked with this project folder as its argument.
BatchManager::initializeExecutions() reads executionSettings.csv,
creates per-column execution folders, and copies data.
"""

import os
import shutil
import argparse

from config import (
    RUNS_DIR, PREPROCESSED_DIR, CSV_SEP, FLOAT_FMT,
    CV_SEEDS, N_FOLDS, DATASETS, LARGE_DATASET_THRESHOLD,
    DATASET_NAMES, ENSEMBLE_K,
)
from settings_generator import generate_ensemble_settings


def get_fold_count(dataset_name):
    """Return number of folds for a dataset (1 for large, N_FOLDS for standard)."""
    if DATASETS[dataset_name]["n"] > LARGE_DATASET_THRESHOLD:
        return 1
    return N_FOLDS


def build_project(dataset_name, seed, template_path, output_base=None):
    """Build a single BatchManager-compatible project folder.

    Args:
        dataset_name:  Dataset name matching fold_generator output.
        seed:          CV seed (determines which fold data to use).
        template_path: Path to base executionSettings.csv template.
        output_base:   Base directory for project folders.
                       Default: RUNS_DIR/projects/

    Returns:
        Path to the created project folder.
    """
    if output_base is None:
        output_base = os.path.join(RUNS_DIR, "projects")

    project_dir = os.path.join(output_base, f"project_{dataset_name}_seed{seed}")
    n_folds = get_fold_count(dataset_name)

    # Generate executionSettings.csv with K ensemble columns.
    # Override output neuron count to match dataset class count.
    n_classes = DATASETS[dataset_name]["classes"]
    settings_path = os.path.join(project_dir, "executionSettings.csv")
    generate_ensemble_settings(
        template_path, settings_path, dataset_name, base_seed=seed, k=ENSEMBLE_K,
        n_classes=n_classes)

    # Copy fold data into DataPool structure.
    for fold_idx in range(n_folds):
        fold_src = os.path.join(
            RUNS_DIR, f"seed_{seed}", f"fold_{fold_idx}", dataset_name)

        # BatchManager expects Fold-N (1-indexed).
        fold_dst = os.path.join(
            project_dir, "DataPool", dataset_name, "Dataset", f"Fold-{fold_idx + 1}")

        os.makedirs(fold_dst, exist_ok=True)

        for csv_name in ["TrainF.csv", "TrainL.csv", "TestF.csv", "TestL.csv"]:
            src = os.path.join(fold_src, csv_name)
            dst = os.path.join(fold_dst, csv_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                raise FileNotFoundError(
                    f"Missing {csv_name} in {fold_src}. Run fold_generator.py first.")

    return project_dir


def build_all_projects(datasets, seeds, template_path, output_base=None):
    """Build project folders for all dataset × seed combinations.

    Returns:
        list of (dataset_name, seed, project_dir) tuples.
    """
    projects = []
    for ds in datasets:
        for seed in seeds:
            project_dir = build_project(ds, seed, template_path, output_base)
            projects.append((ds, seed, project_dir))
    return projects


def main():
    parser = argparse.ArgumentParser(
        description="Build BatchManager-compatible project folders")
    parser.add_argument("--template", required=True,
                        help="Path to base executionSettings.csv template")
    parser.add_argument("--dataset", default=None,
                        help="Single dataset name")
    parser.add_argument("--all", action="store_true",
                        help="Build for all 28 datasets")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help=f"Seeds (default: {CV_SEEDS})")
    parser.add_argument("--output-base", default=None,
                        help="Base directory for project folders")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        print("ERROR: Specify --dataset <name> or --all")
        return

    datasets = [args.dataset] if args.dataset else DATASET_NAMES
    seeds = args.seeds if args.seeds else CV_SEEDS

    print(f"Project Builder | K={ENSEMBLE_K} ensemble members")
    print(f"  Template: {args.template}")
    print(f"  Seeds: {seeds}")
    print(f"{'=' * 60}")

    for ds in datasets:
        if ds not in DATASETS:
            print(f"  [{ds}] SKIPPED — not in dataset registry")
            continue

        for seed in seeds:
            print(f"  [{ds}] seed={seed} ", end="", flush=True)
            try:
                n_folds = get_fold_count(ds)
                project_dir = build_project(
                    ds, seed, args.template, args.output_base)
                print(f"OK  ({n_folds} folds, {project_dir})")
            except Exception as e:
                print(f"FAILED  ({e})")

    print("\nDone.")


if __name__ == "__main__":
    main()
