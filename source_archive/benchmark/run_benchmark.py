#!/usr/bin/env python3
"""
run_benchmark.py — Main orchestrator for the DEBI-NN benchmark harness.

Stages (run individually or all at once):
    1. generate-folds   — Create stratified CV folds from preprocessed data
    2. build-projects   — Assemble BatchManager project folders
    3. run-debinn       — Invoke C++ binary on each project folder
    4. run-competitors  — Run competitor baselines on same folds
    5. aggregate        — Ensemble averaging + collect all results
    6. analyze          — Statistical analysis + tables + LaTeX

Usage:
    python run_benchmark.py --stage all --template base_settings.csv
    python run_benchmark.py --stage generate-folds
    python run_benchmark.py --stage run-debinn --dataset iris
    python run_benchmark.py --stage analyze --metric BACC
"""

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import (
    RUNS_DIR, RESULTS_DIR, DEBINN_BINARY,
    CV_SEEDS, N_FOLDS, DATASETS, LARGE_DATASET_THRESHOLD,
    DATASET_NAMES, ENSEMBLE_K, COMPETITORS, DEBINN_NUMA_NODES,
)


def stage_generate_folds(datasets, seeds):
    """Stage 1: Generate stratified CV folds."""
    from fold_generator import generate_splits_for_dataset

    print(f"\n{'='*60}")
    print(f"STAGE 1: Generate Folds | {len(datasets)} datasets × {len(seeds)} seeds")
    print(f"{'='*60}")

    for ds in datasets:
        print(f"  [{ds}] ", end="", flush=True)
        try:
            results, is_large = generate_splits_for_dataset(ds, seeds)
            tag = "LARGE" if is_large else f"{N_FOLDS}-fold"
            print(f"OK ({tag})")
        except Exception as e:
            print(f"FAILED ({e})")


def stage_build_projects(datasets, seeds, template_path):
    """Stage 2: Build BatchManager project folders."""
    from project_builder import build_project

    print(f"\n{'='*60}")
    print(f"STAGE 2: Build Projects | template={template_path}")
    print(f"{'='*60}")

    if not os.path.exists(template_path):
        print(f"ERROR: Template not found: {template_path}")
        return

    for ds in datasets:
        for seed in seeds:
            print(f"  [{ds}] seed={seed} ", end="", flush=True)
            try:
                project_dir = build_project(ds, seed, template_path)
                print(f"OK ({project_dir})")
            except Exception as e:
                print(f"FAILED ({e})")


def _run_single_debinn(args_tuple):
    """Worker for parallel DEBI-NN runs."""
    from debinn_runner import run_debinn
    ds, seed, project_dir, binary, numa_node = args_tuple
    result = run_debinn(project_dir, binary, numa_node=numa_node)
    return ds, seed, result


def stage_run_debinn(datasets, seeds, binary_path, max_workers=1):
    """Stage 3: Run DEBI-NN on project folders."""
    from debinn_runner import run_debinn, verify_outputs

    numa_nodes = DEBINN_NUMA_NODES
    numa_tag = f"NUMA={numa_nodes} nodes" if numa_nodes > 0 else "NUMA=off"

    print(f"\n{'='*60}")
    print(f"STAGE 3: Run DEBI-NN | binary={binary_path} | workers={max_workers} | {numa_tag}")
    print(f"{'='*60}")

    if not os.path.exists(binary_path):
        print(f"ERROR: Binary not found: {binary_path}")
        return

    tasks = []
    task_idx = 0
    for ds in datasets:
        for seed in seeds:
            project_dir = os.path.join(
                RUNS_DIR, "projects", f"project_{ds}_seed{seed}")
            if not os.path.isdir(project_dir):
                print(f"  [{ds}] seed={seed} SKIPPED — project folder missing")
                continue
            numa_node = (task_idx % numa_nodes) if numa_nodes > 0 else None
            tasks.append((ds, seed, project_dir, binary_path, numa_node))
            task_idx += 1

    t0 = time.time()

    if max_workers <= 1:
        # Sequential — safer, less RAM.
        for ds, seed, project_dir, binary, numa_node in tasks:
            print(f"  [{ds}] seed={seed} ", end="", flush=True)
            result = run_debinn(project_dir, binary, numa_node=numa_node)
            if result["success"]:
                is_large = DATASETS[ds]["n"] > LARGE_DATASET_THRESHOLD
                n_folds = 1 if is_large else N_FOLDS
                found, missing = verify_outputs(
                    project_dir, ds, n_folds, ENSEMBLE_K)
                print(f"OK ({result['elapsed_sec']:.0f}s, "
                      f"{len(found)}/{len(found)+len(missing)} predictions)")
            else:
                print(f"FAILED (rc={result['returncode']}, "
                      f"{result['elapsed_sec']:.0f}s)")
    else:
        # Parallel across datasets (not within — BatchManager is sequential).
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_single_debinn, t): t for t in tasks}
            for future in as_completed(futures):
                ds, seed, result = future.result()
                status = "OK" if result["success"] else "FAILED"
                print(f"  [{ds}] seed={seed} {status} "
                      f"({result['elapsed_sec']:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  Total DEBI-NN time: {elapsed/60:.1f} minutes")


def stage_run_competitors(datasets, seeds, methods=None, max_workers=4):
    """Stage 4: Run competitor baselines."""
    from competitor_runner import run_all_competitors, get_fold_count

    if methods is None:
        methods = COMPETITORS

    print(f"\n{'='*60}")
    print(f"STAGE 4: Run Competitors | methods={methods} | workers={max_workers}")
    print(f"{'='*60}")

    t0 = time.time()

    for ds in datasets:
        n_folds = get_fold_count(ds)
        for seed in seeds:
            for fold_idx in range(n_folds):
                print(f"  [{ds}] seed={seed} fold={fold_idx} ", end="", flush=True)
                results = run_all_competitors(ds, seed, fold_idx, methods)

                parts = []
                for m, r in results.items():
                    if "error" in r:
                        parts.append(f"{m}:SKIP")
                    else:
                        parts.append(f"{m}:{r['BACC']:.3f}")
                print("  ".join(parts))

    elapsed = time.time() - t0
    print(f"\n  Total competitor time: {elapsed/60:.1f} minutes")


def stage_aggregate(datasets, seeds):
    """Stage 5: Ensemble aggregation + results collection."""
    from ensemble_aggregator import aggregate_dataset_seed
    from results_collector import (
        collect_debinn_results, collect_competitor_results,
    )

    print(f"\n{'='*60}")
    print(f"STAGE 5: Aggregate Results")
    print(f"{'='*60}")

    # Ensemble aggregation.
    print("  Ensemble aggregation...")
    for ds in datasets:
        for seed in seeds:
            print(f"    [{ds}] seed={seed} ", end="", flush=True)
            try:
                fold_results = aggregate_dataset_seed(ds, seed)
                ens_bacc = sum(r["ensemble"]["BACC"] for r in fold_results) / len(fold_results)
                print(f"OK (ensemble BACC={ens_bacc:.3f})")
            except Exception as e:
                print(f"SKIP ({e})")

    # Collect all results.
    print("\n  Collecting all results...")
    import pandas as pd
    debinn_rows = collect_debinn_results(datasets, seeds)
    comp_rows = collect_competitor_results(datasets, seeds)
    all_rows = debinn_rows + comp_rows

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.sort_values(["dataset", "method", "seed", "fold"]).reset_index(drop=True)
        output_path = os.path.join(RESULTS_DIR, "results_table.csv")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Results table: {output_path}")
        print(f"  {len(df)} rows, {df['method'].nunique()} methods, "
              f"{df['dataset'].nunique()} datasets")
    else:
        print("  No results found.")


def stage_analyze(metric="BACC"):
    """Stage 6: Statistical analysis."""
    from analysis import main as analysis_main

    print(f"\n{'='*60}")
    print(f"STAGE 6: Analysis | metric={metric}")
    print(f"{'='*60}")

    # Invoke analysis.main() with constructed args.
    sys.argv = [
        "analysis.py",
        "--metric", metric,
    ]
    analysis_main()


def main():
    parser = argparse.ArgumentParser(
        description="DEBI-NN Benchmark Harness — Main Orchestrator")
    parser.add_argument("--stage", required=True,
                        choices=["all", "generate-folds", "build-projects",
                                 "run-debinn", "run-competitors",
                                 "aggregate", "analyze"],
                        help="Which stage to run")
    parser.add_argument("--template", default=None,
                        help="Base executionSettings.csv template (required for build-projects)")
    parser.add_argument("--dataset", default=None,
                        help="Single dataset (default: all 28)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help=f"CV seeds (default: {CV_SEEDS})")
    parser.add_argument("--binary", default=None,
                        help=f"DEBI-NN binary path (default: {DEBINN_BINARY})")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Competitor methods (default: all)")
    parser.add_argument("--metric", default="BACC",
                        help="Primary metric for analysis (default: BACC)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for DEBI-NN/competitor runs")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASET_NAMES
    seeds = args.seeds if args.seeds else CV_SEEDS
    binary = args.binary or DEBINN_BINARY

    # Validate datasets.
    datasets = [d for d in datasets if d in DATASETS]

    stage = args.stage

    if stage in ("all", "generate-folds"):
        stage_generate_folds(datasets, seeds)

    if stage in ("all", "build-projects"):
        template = args.template
        if not template:
            print("ERROR: --template required for build-projects stage")
            if stage != "all":
                return
        else:
            stage_build_projects(datasets, seeds, template)

    if stage in ("all", "run-debinn"):
        stage_run_debinn(datasets, seeds, binary, args.workers)

    if stage in ("all", "run-competitors"):
        stage_run_competitors(datasets, seeds, args.methods, args.workers)

    if stage in ("all", "aggregate"):
        stage_aggregate(datasets, seeds)

    if stage in ("all", "analyze"):
        stage_analyze(args.metric)

    print("\nBenchmark harness complete.")


if __name__ == "__main__":
    main()
