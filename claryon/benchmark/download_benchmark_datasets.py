"""Download 12 curated medical benchmark datasets for CLARYON evaluation.

Sources: OpenML, UCI Machine Learning Repository, Kaggle.
All datasets are medical/biomedical classification tasks.

Usage:
    python -m claryon.benchmark.download_benchmark_datasets [--output-dir benchmark_data]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Dataset registry — 12 medical datasets
# ─────────────────────────────────────────────

OPENML_DATASETS: List[Dict[str, Any]] = [
    {"name": "blood-transfusion",       "openml_id": 1464, "target": "Class",    "tier": 1},
    {"name": "diabetes",                "openml_id": 37,   "target": "class",    "tier": 1},
    {"name": "iris",                    "openml_id": 61,   "target": "class",    "tier": 0},  # demo only
]

UCI_DATASETS: List[Dict[str, Any]] = [
    {"name": "wisconsin-breast-cancer", "uci_id": 17,   "tier": 3},
    {"name": "heart-failure",           "uci_id": 519,  "tier": 3},
    {"name": "cervical-cancer",         "uci_id": 383,  "tier": 3},
    {"name": "chronic-kidney-disease",  "uci_id": 336,  "tier": 3},
    {"name": "spect-heart",             "uci_id": 95,   "tier": 3},
    {"name": "mammographic-mass",       "uci_id": 161,  "tier": 3},
]

KAGGLE_DATASETS: List[Dict[str, Any]] = [
    {"name": "hcc-survival",            "slug": "mrsantos/hcc-dataset",                    "tier": 3},
    {"name": "stroke-prediction",       "slug": "fedesoriano/stroke-prediction-dataset",   "tier": 3},
    {"name": "fetal-health",            "slug": "andrewmvd/fetal-health-classification",   "tier": 3},
]

TOTAL_DATASETS = len(OPENML_DATASETS) + len(UCI_DATASETS) + len(KAGGLE_DATASETS)


def download_openml(dataset: Dict[str, Any], output_dir: Path) -> bool:
    """Download a dataset from OpenML.

    Args:
        dataset: Dataset entry with openml_id and target.
        output_dir: Directory to save the CSV.

    Returns:
        True if successful.
    """
    try:
        from sklearn.datasets import fetch_openml
        import pandas as pd

        name = dataset["name"]
        ds = fetch_openml(data_id=dataset["openml_id"], as_frame=True, parser="auto")
        df = ds.frame
        if df is None:
            logger.error("OpenML returned empty frame for %s", name)
            return False

        out_path = output_dir / name / f"{name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep=";", index=False)
        logger.info("Downloaded %s: %d samples × %d features → %s",
                     name, len(df), len(df.columns) - 1, out_path)
        return True
    except Exception as e:
        logger.error("Failed to download %s from OpenML: %s", dataset["name"], e)
        return False


def download_uci(dataset: Dict[str, Any], output_dir: Path) -> bool:
    """Download a dataset from UCI ML Repository.

    Args:
        dataset: Dataset entry with uci_id.
        output_dir: Directory to save the CSV.

    Returns:
        True if successful.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        import pandas as pd

        name = dataset["name"]
        ds = fetch_ucirepo(id=dataset["uci_id"])
        X = ds.data.features
        y = ds.data.targets
        df = pd.concat([X, y], axis=1)

        out_path = output_dir / name / f"{name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep=";", index=False)
        logger.info("Downloaded %s: %d samples × %d features → %s",
                     name, len(df), len(df.columns) - 1, out_path)
        return True
    except Exception as e:
        logger.error("Failed to download %s from UCI: %s", dataset["name"], e)
        return False


def download_kaggle(dataset: Dict[str, Any], output_dir: Path) -> bool:
    """Download a dataset from Kaggle.

    Requires kaggle CLI configured with API credentials.

    Args:
        dataset: Dataset entry with slug.
        output_dir: Directory to save files.

    Returns:
        True if successful.
    """
    try:
        import subprocess

        name = dataset["name"]
        dest = output_dir / name
        dest.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset["slug"],
             "-p", str(dest), "--unzip"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error("Kaggle download failed for %s: %s", name, result.stderr)
            return False
        logger.info("Downloaded %s from Kaggle → %s", name, dest)
        return True
    except Exception as e:
        logger.error("Failed to download %s from Kaggle: %s", dataset["name"], e)
        return False


def download_all(output_dir: str = "benchmark_data") -> Dict[str, bool]:
    """Download all 12 benchmark datasets.

    Args:
        output_dir: Root directory for downloaded datasets.

    Returns:
        Dict mapping dataset name to success status.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: Dict[str, bool] = {}

    logger.info("Downloading %d benchmark datasets to %s", TOTAL_DATASETS, out)

    for ds in OPENML_DATASETS:
        results[ds["name"]] = download_openml(ds, out)

    for ds in UCI_DATASETS:
        results[ds["name"]] = download_uci(ds, out)

    for ds in KAGGLE_DATASETS:
        results[ds["name"]] = download_kaggle(ds, out)

    ok = sum(1 for v in results.values() if v)
    logger.info("Downloaded %d/%d datasets successfully", ok, TOTAL_DATASETS)
    return results


def main() -> None:
    """CLI entry point for benchmark dataset download."""
    parser = argparse.ArgumentParser(
        description="Download 12 medical benchmark datasets for CLARYON"
    )
    parser.add_argument(
        "--output-dir", default="benchmark_data",
        help="Output directory (default: benchmark_data)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity",
    )
    args = parser.parse_args()

    level = logging.WARNING
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose >= 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    download_all(args.output_dir)


if __name__ == "__main__":
    main()
