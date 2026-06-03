"""FDB/LDB legacy format loader — radiomics compatible format → Dataset.

Ported from [E] build_tabular_from_fdb_ldb.py and [B] fold_generator.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .base import Dataset, MultiClassLabelMapper, BinaryLabelMapper, TaskType

logger = logging.getLogger(__name__)

SEP = ";"


def load_fdb_ldb(
    fdb_path: Union[str, Path],
    ldb_path: Union[str, Path],
    task_type: Optional[TaskType] = None,
) -> Dataset:
    """Load FDB (feature database) and LDB (label database) files into a Dataset.

    Args:
        fdb_path: Path to FDB.csv (Key;F0;F1;...;FN, semicolon-separated).
        ldb_path: Path to LDB.csv (Key;Label, semicolon-separated).
        task_type: Override task type. If None, inferred from labels.

    Returns:
        Dataset with features, labels, keys, and feature names.

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If FDB/LDB Key columns don't match.
    """
    fdb_path = Path(fdb_path)
    ldb_path = Path(ldb_path)

    if not fdb_path.exists():
        raise FileNotFoundError(f"FDB file not found: {fdb_path}")
    if not ldb_path.exists():
        raise FileNotFoundError(f"LDB file not found: {ldb_path}")

    fdb = pd.read_csv(fdb_path, sep=SEP)
    ldb = pd.read_csv(ldb_path, sep=SEP)

    if len(fdb) != len(ldb):
        raise ValueError(
            f"FDB/LDB row count mismatch: {len(fdb)} vs {len(ldb)}"
        )

    # Extract keys
    key_col = "Key" if "Key" in fdb.columns else fdb.columns[0]
    keys_fdb = fdb[key_col].astype(str).tolist()
    keys_ldb = ldb[key_col].astype(str).tolist()
    if keys_fdb != keys_ldb:
        raise ValueError("FDB/LDB Key column values do not match")

    keys = keys_fdb

    # Extract features
    feature_cols = [c for c in fdb.columns if c != key_col]
    feature_names = list(feature_cols)
    X = fdb[feature_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Extract labels
    label_col = "Label" if "Label" in ldb.columns else [c for c in ldb.columns if c != key_col][0]
    y_raw = ldb[label_col]

    if task_type is None:
        n_unique = y_raw.nunique()
        task_type = TaskType.BINARY if n_unique == 2 else TaskType.MULTICLASS

    if task_type == TaskType.BINARY:
        label_mapper = BinaryLabelMapper.fit(y_raw.values)
    else:
        label_mapper = MultiClassLabelMapper.fit(y_raw.values)

    y = label_mapper.transform(y_raw.values)

    return Dataset(
        X=X,
        y=y,
        keys=keys,
        feature_names=feature_names,
        task_type=task_type,
        label_mapper=label_mapper,
    )


def write_fdb_ldb(
    dataset: Dataset,
    fdb_path: Union[str, Path],
    ldb_path: Union[str, Path],
) -> None:
    """Write a Dataset to FDB/LDB format.

    Args:
        dataset: Dataset to write.
        fdb_path: Output path for FDB.csv.
        ldb_path: Output path for LDB.csv.
    """
    fdb_path = Path(fdb_path)
    ldb_path = Path(ldb_path)
    fdb_path.parent.mkdir(parents=True, exist_ok=True)
    ldb_path.parent.mkdir(parents=True, exist_ok=True)

    keys = dataset.keys or [f"S{i:04d}" for i in range(dataset.n_samples)]
    feature_names = dataset.feature_names or [f"F{i}" for i in range(dataset.n_features)]

    # FDB
    fdb = pd.DataFrame(dataset.X, columns=feature_names)
    fdb.insert(0, "Key", keys)
    fdb.to_csv(fdb_path, sep=SEP, index=False, float_format="%.8f")

    # LDB
    if dataset.y is not None:
        ldb = pd.DataFrame({"Key": keys, "Label": dataset.y})
        ldb.to_csv(ldb_path, sep=SEP, index=False)
    else:
        ldb = pd.DataFrame({"Key": keys, "Label": [""] * len(keys)})
        ldb.to_csv(ldb_path, sep=SEP, index=False)

    logger.debug("Wrote FDB to %s, LDB to %s", fdb_path, ldb_path)
