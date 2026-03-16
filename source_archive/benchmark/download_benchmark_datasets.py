#!/usr/bin/env python3
"""
download_benchmark_datasets.py — DEBI-NN Benchmark Dataset Downloader

Downloads all 28 datasets specified in DEBINN_BENCHMARK_PROTOCOL_E8.md.
Saves each dataset as CSV files in a structured directory.

Usage:
    pip install openml scikit-learn ucimlrepo kaggle pandas
    python download_benchmark_datasets.py [--output-dir benchmarking/datasets]

Requirements:
    - openml: for OpenML datasets
    - ucimlrepo: for UCI datasets
    - kaggle: for Kaggle datasets (requires ~/.kaggle/kaggle.json API token)
    - pandas, scikit-learn: for data handling
"""

import os
import sys
import json
import argparse
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Dataset Registry
# ──────────────────────────────────────────────────────────────────────

OPENML_DATASETS = [
    # Tier 1: OpenML Standard
    {"name": "australian",              "openml_id": 14,   "target": "A15",            "tier": 1},
    {"name": "blood-transfusion",       "openml_id": 1464, "target": "Class",          "tier": 1},
    {"name": "credit-g",                "openml_id": 31,   "target": "class",          "tier": 1},
    {"name": "diabetes",                "openml_id": 37,   "target": "class",          "tier": 1},
    {"name": "kc1",                     "openml_id": 1067, "target": "defects",        "tier": 1},
    {"name": "phoneme",                 "openml_id": 1489, "target": "Class",          "tier": 1},
    # Tier 2: Additional Standard
    {"name": "iris",                    "openml_id": 61,   "target": "class",          "tier": 2},
    {"name": "vehicle",                 "openml_id": 54,   "target": "Class",          "tier": 2},
    {"name": "segment",                 "openml_id": 36,   "target": "class",          "tier": 2},
    {"name": "waveform-5000",           "openml_id": 60,   "target": "class",          "tier": 2},
    {"name": "steel-plates-fault",      "openml_id": 1504, "target": "target",         "tier": 2},
    {"name": "electricity",             "openml_id": 151,  "target": "class",          "tier": 2},
    {"name": "bank-marketing",          "openml_id": 1461, "target": "Class",          "tier": 2},
    {"name": "adult",                   "openml_id": 1590, "target": "class",          "tier": 2},
]

UCI_DATASETS = [
    # Tier 3: Medical
    {"name": "wisconsin-breast-cancer", "uci_id": 17,   "tier": 3},
    {"name": "heart-failure",           "uci_id": 519,  "tier": 3},
    {"name": "cervical-cancer",         "uci_id": 383,  "tier": 3},
    {"name": "chronic-kidney-disease",  "uci_id": 336,  "tier": 3},
    {"name": "spect-heart",             "uci_id": 95,   "tier": 3},
    {"name": "mammographic-mass",       "uci_id": 161,  "tier": 3},
    # Tier 4: General domain
    {"name": "wine-quality",            "uci_id": 186,  "tier": 4},
    {"name": "dry-bean",                "uci_id": 602,  "tier": 4},
    {"name": "rice-cammeo-osmancik",    "uci_id": 545,  "tier": 4},
    {"name": "mushroom",                "uci_id": 73,   "tier": 4},
]

KAGGLE_DATASETS = [
    # Tier 3: Medical
    {"name": "hcc-survival",            "slug": "mrsantos/hcc-dataset",                    "tier": 3},
    {"name": "stroke-prediction",       "slug": "fedesoriano/stroke-prediction-dataset",   "tier": 3},
    # Tier 4: General domain
    {"name": "drug-classification",     "slug": "prathamtripathi/drug-classification",      "tier": 4},
    {"name": "fetal-health",            "slug": "andrewmvd/fetal-health-classification",   "tier": 4},
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def download_openml_datasets(output_dir):
    """Download all OpenML datasets."""
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        return []

    results = []
    for ds in OPENML_DATASETS:
        name = ds["name"]
        did = ds["openml_id"]
        ds_dir = ensure_dir(os.path.join(output_dir, name))

        print(f"  [{name}] OpenML ID={did} ... ", end="", flush=True)
        try:
            data = fetch_openml(data_id=did, as_frame=True, parser="auto")
            df = data.frame

            if df is None:
                print("FAILED (no frame returned)")
                continue

            target_col = data.target_names[0] if data.target_names else ds.get("target", "target")

            # Separate features and target
            if target_col in df.columns:
                X = df.drop(columns=[target_col])
                y = df[[target_col]]
            else:
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1:]
                target_col = y.columns[0]

            X.to_csv(os.path.join(ds_dir, "features.csv"), index=False)
            y.to_csv(os.path.join(ds_dir, "labels.csv"), index=False)

            info = {
                "name": name,
                "source": "OpenML",
                "openml_id": did,
                "n_samples": len(df),
                "n_features": X.shape[1],
                "n_classes": y[target_col].nunique(),
                "target_column": target_col,
                "tier": ds["tier"],
                "class_distribution": y[target_col].value_counts().to_dict(),
            }
            with open(os.path.join(ds_dir, "info.json"), "w") as f:
                json.dump(info, f, indent=2, default=str)

            print(f"OK ({info['n_samples']} samples, {info['n_features']} features, {info['n_classes']} classes)")
            results.append(info)

        except Exception as e:
            print(f"FAILED ({e})")

    return results


def download_uci_datasets(output_dir):
    """Download all UCI datasets."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("ERROR: ucimlrepo not installed. Run: pip install ucimlrepo")
        return []

    results = []
    for ds in UCI_DATASETS:
        name = ds["name"]
        uid = ds["uci_id"]
        ds_dir = ensure_dir(os.path.join(output_dir, name))

        print(f"  [{name}] UCI ID={uid} ... ", end="", flush=True)
        try:
            data = fetch_ucirepo(id=uid)
            X = data.data.features
            y = data.data.targets

            X.to_csv(os.path.join(ds_dir, "features.csv"), index=False)
            y.to_csv(os.path.join(ds_dir, "labels.csv"), index=False)

            target_col = y.columns[0]
            info = {
                "name": name,
                "source": "UCI",
                "uci_id": uid,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": int(y[target_col].nunique()),
                "target_column": target_col,
                "tier": ds["tier"],
                "class_distribution": y[target_col].value_counts().to_dict(),
            }
            with open(os.path.join(ds_dir, "info.json"), "w") as f:
                json.dump(info, f, indent=2, default=str)

            print(f"OK ({info['n_samples']} samples, {info['n_features']} features, {info['n_classes']} classes)")
            results.append(info)

        except Exception as e:
            print(f"FAILED ({e})")

    return results


def download_kaggle_datasets(output_dir):
    """Download all Kaggle datasets. Requires kaggle CLI and API token."""
    results = []

    try:
        import kaggle
    except (ImportError, OSError) as e:
        print(f"WARNING: Kaggle not available ({e}). Skipping Kaggle datasets.")
        print("  To enable: pip install kaggle && set up ~/.kaggle/kaggle.json")
        return results

    import zipfile
    import tempfile

    for ds in KAGGLE_DATASETS:
        name = ds["name"]
        slug = ds["slug"]
        ds_dir = ensure_dir(os.path.join(output_dir, name))

        print(f"  [{name}] Kaggle: {slug} ... ", end="", flush=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                kaggle.api.dataset_download_files(slug, path=tmp, unzip=True)

                # Find CSV files in the downloaded directory
                csv_files = []
                for root, dirs, files in os.walk(tmp):
                    for f in files:
                        if f.endswith(".csv"):
                            csv_files.append(os.path.join(root, f))

                if not csv_files:
                    print("FAILED (no CSV files found)")
                    continue

                # Use the largest CSV file as the primary dataset
                main_csv = max(csv_files, key=os.path.getsize)
                df = pd.read_csv(main_csv)

                # Save the raw combined file
                df.to_csv(os.path.join(ds_dir, "raw.csv"), index=False)

                info = {
                    "name": name,
                    "source": "Kaggle",
                    "kaggle_slug": slug,
                    "n_samples": len(df),
                    "n_columns": df.shape[1],
                    "columns": list(df.columns),
                    "tier": ds["tier"],
                    "note": "Raw download. Manual inspection needed to split features/labels.",
                }
                with open(os.path.join(ds_dir, "info.json"), "w") as f:
                    json.dump(info, f, indent=2, default=str)

                print(f"OK ({info['n_samples']} samples, {info['n_columns']} columns)")
                results.append(info)

        except Exception as e:
            print(f"FAILED ({e})")

    return results


def generate_manifest(output_dir, all_results):
    """Generate a manifest file summarizing all downloaded datasets."""
    manifest_path = os.path.join(output_dir, "MANIFEST.json")
    manifest = {
        "protocol_version": "E8 v1.0",
        "date": "2026-03-05",
        "total_datasets": len(all_results),
        "datasets": all_results,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest written to {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Download DEBI-NN benchmark datasets")
    parser.add_argument("--output-dir", default="benchmarking/datasets",
                        help="Output directory for downloaded datasets")
    parser.add_argument("--skip-kaggle", action="store_true",
                        help="Skip Kaggle datasets (requires API token)")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    print(f"DEBI-NN Benchmark Dataset Downloader")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"=" * 60)

    all_results = []

    print(f"\n[1/3] Downloading OpenML datasets ({len(OPENML_DATASETS)} datasets)...")
    all_results.extend(download_openml_datasets(output_dir))

    print(f"\n[2/3] Downloading UCI datasets ({len(UCI_DATASETS)} datasets)...")
    all_results.extend(download_uci_datasets(output_dir))

    if not args.skip_kaggle:
        print(f"\n[3/3] Downloading Kaggle datasets ({len(KAGGLE_DATASETS)} datasets)...")
        all_results.extend(download_kaggle_datasets(output_dir))
    else:
        print(f"\n[3/3] Skipping Kaggle datasets (--skip-kaggle)")

    print(f"\n{'=' * 60}")
    print(f"Downloaded: {len(all_results)} / 28 datasets")
    generate_manifest(output_dir, all_results)

    # Summary table
    print(f"\nDataset Summary:")
    print(f"{'Name':<30} {'Source':<8} {'N':>8} {'Feat':>6} {'Cls':>5} {'Tier':>5}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*6} {'-'*5} {'-'*5}")
    for r in sorted(all_results, key=lambda x: x.get("tier", 0)):
        n_feat = r.get("n_features", r.get("n_columns", "?"))
        n_cls = r.get("n_classes", "?")
        print(f"{r['name']:<30} {r['source']:<8} {r['n_samples']:>8} {n_feat:>6} {n_cls!s:>5} {r['tier']:>5}")


if __name__ == "__main__":
    main()
