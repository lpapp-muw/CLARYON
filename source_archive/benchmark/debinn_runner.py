#!/usr/bin/env python3
"""
debinn_runner.py — Invoke the DEBI-NN C++ binary via subprocess.

The binary is treated as an opaque, protected executable.
Interface is entirely file-based:
    Input:  project_folder (CLI arg)
    Output: Executions-Finished/{dataset}-M{k}/Log/Fold-{N}/Predictions.csv
"""

import os
import subprocess
import time
import argparse

from config import (
    DEBINN_BINARY, DEBINN_TIMEOUT_SEC, DEBINN_OMP_THREADS, DEBINN_NUMA_NODES,
    RUNS_DIR, CV_SEEDS, DATASET_NAMES,
)


def run_debinn(project_dir, binary_path=None, timeout=None, numa_node=None):
    """Invoke the DEBI-NN binary on a project folder.

    Args:
        project_dir:  Path to BatchManager-compatible project folder.
        binary_path:  Path to the DEBI-NN executable (default: config.DEBINN_BINARY).
        timeout:      Timeout in seconds (default: config.DEBINN_TIMEOUT_SEC).
        numa_node:    NUMA node to pin to (None = use round-robin from config).

    Returns:
        dict with keys: success, returncode, elapsed_sec, stdout, stderr.
    """
    if binary_path is None:
        binary_path = DEBINN_BINARY
    if timeout is None:
        timeout = DEBINN_TIMEOUT_SEC

    if not os.path.exists(binary_path):
        return {
            "success": False,
            "returncode": -1,
            "elapsed_sec": 0.0,
            "stdout": "",
            "stderr": f"Binary not found: {binary_path}",
        }

    if not os.path.isdir(project_dir):
        return {
            "success": False,
            "returncode": -1,
            "elapsed_sec": 0.0,
            "stdout": "",
            "stderr": f"Project folder not found: {project_dir}",
        }

    # Ensure trailing slash (BatchManager expects it).
    proj = project_dir.rstrip("/") + "/"

    # Build command with optional NUMA pinning.
    if numa_node is not None and numa_node >= 0:
        cmd = ["numactl", f"--cpunodebind={numa_node}", f"--membind={numa_node}",
               binary_path, proj]
    else:
        cmd = [binary_path, proj]

    # Set OMP thread count and Qt env in subprocess environment.
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(DEBINN_OMP_THREADS)
    env["QT_QPA_PLATFORM"] = "offscreen"

    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_dir,
            env=env,
        )
        elapsed = time.time() - t0

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "elapsed_sec": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {
            "success": False,
            "returncode": -999,
            "elapsed_sec": elapsed,
            "stdout": "",
            "stderr": f"TIMEOUT after {timeout}s",
        }

    except Exception as e:
        elapsed = time.time() - t0
        return {
            "success": False,
            "returncode": -1,
            "elapsed_sec": elapsed,
            "stdout": "",
            "stderr": str(e),
        }


def verify_outputs(project_dir, dataset_name, n_folds, k):
    """Check that expected Predictions.csv files exist after a run.

    Returns:
        (found, missing): lists of paths.
    """
    finished_dir = os.path.join(project_dir, "Executions-Finished")
    found = []
    missing = []

    for member_idx in range(k):
        exec_name = f"{dataset_name}-M{member_idx}"
        for fold_idx in range(n_folds):
            pred_path = os.path.join(
                finished_dir, exec_name, "Log",
                f"Fold-{fold_idx + 1}", "Predictions.csv")
            if os.path.exists(pred_path):
                found.append(pred_path)
            else:
                missing.append(pred_path)

    return found, missing


def main():
    parser = argparse.ArgumentParser(
        description="Run DEBI-NN binary on project folders")
    parser.add_argument("--project", default=None,
                        help="Single project folder to run")
    parser.add_argument("--binary", default=None,
                        help=f"Path to DEBI-NN binary (default: {DEBINN_BINARY})")
    parser.add_argument("--timeout", type=int, default=None,
                        help=f"Timeout in seconds (default: {DEBINN_TIMEOUT_SEC})")
    args = parser.parse_args()

    if not args.project:
        print("ERROR: Specify --project <path>")
        return

    print(f"Running DEBI-NN: {args.project}")
    result = run_debinn(args.project, args.binary, args.timeout)

    if result["success"]:
        print(f"  OK ({result['elapsed_sec']:.1f}s)")
    else:
        print(f"  FAILED (rc={result['returncode']}, {result['elapsed_sec']:.1f}s)")
        if result["stderr"]:
            # Print last 20 lines of stderr.
            lines = result["stderr"].strip().split("\n")
            for line in lines[-20:]:
                print(f"    {line}")


if __name__ == "__main__":
    main()
