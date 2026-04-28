"""Center label loader — maps case IDs to center identifiers."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def load_center_labels(csv_path: Path, sep: str = ";") -> Dict[str, str]:
    """Read a center-mapping CSV (columns: case_id, center).

    Args:
        csv_path: Path to the CSV file.
        sep: Column separator (default ";").

    Returns:
        Dict mapping case_id → center string.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Center labels file not found: {csv_path}")

    mapping: Dict[str, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        fieldnames = reader.fieldnames or []
        if "case_id" not in fieldnames or "center" not in fieldnames:
            raise ValueError(
                f"Center labels CSV must have 'case_id' and 'center' columns, "
                f"got: {fieldnames}"
            )
        for row in reader:
            mapping[row["case_id"]] = row["center"]

    logger.info("Loaded %d center labels from %s", len(mapping), csv_path)
    return mapping


def attach_center_ids(
    keys: List[str], center_map: Dict[str, str]
) -> np.ndarray:
    """Map dataset keys to center IDs.

    Args:
        keys: Sample identifiers from the dataset.
        center_map: Mapping from case_id to center string.

    Returns:
        Array of center strings, same length as keys.

    Raises:
        ValueError: If any key is not found in center_map.
    """
    missing = [k for k in keys if k not in center_map]
    if missing:
        raise ValueError(
            f"{len(missing)} key(s) not found in center map: {missing[:5]}"
        )
    return np.array([center_map[k] for k in keys])
