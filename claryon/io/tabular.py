"""Tabular data loader — CSV/Parquet → Dataset.

Ported from [E] tabular.py. Encoding decoupled: no amplitude encoding here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .base import BinaryLabelMapper, Dataset, MultiClassLabelMapper, RegressionTarget, TaskType

logger = logging.getLogger(__name__)


def load_tabular_csv(
    path: Union[str, Path],
    label_col: str = "label",
    id_col: Optional[str] = "Key",
    sep: str = ";",
    task_type: Optional[TaskType] = None,
    max_features: Optional[int] = None,
) -> Dataset:
    """Load a tabular CSV into a Dataset.

    Args:
        path: Path to the CSV file.
        label_col: Name of the label/target column.
        id_col: Name of the sample ID column. If None or missing, sequential
            IDs are generated.
        sep: CSV separator character.
        task_type: Override task type detection. If None, inferred from labels.
        max_features: Optional cap on feature count (drops trailing columns).

    Returns:
        Dataset with features, labels, keys, and feature names.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=sep)
    logger.debug("Loaded %s: %d rows × %d cols", path, len(df), len(df.columns))

    # Extract IDs
    if id_col is not None and id_col in df.columns:
        keys = df[id_col].astype(str).tolist()
        df = df.drop(columns=[id_col])
    else:
        keys = [f"S{i:04d}" for i in range(len(df))]

    # Extract labels
    y_raw = None
    label_mapper = None
    if label_col in df.columns:
        y_raw = df[label_col]
        df = df.drop(columns=[label_col])

    # Keep only numeric columns
    Xdf = df.apply(pd.to_numeric, errors="coerce")

    # Optional feature cap
    if max_features is not None and Xdf.shape[1] > max_features:
        Xdf = Xdf.iloc[:, :max_features]

    feature_names = list(Xdf.columns)
    X = Xdf.to_numpy(dtype=np.float64, copy=True)

    # Sanitize NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Determine task type and create label mapper
    y = None
    if y_raw is not None:
        if task_type is None:
            task_type = _infer_task_type(y_raw)
        if task_type == TaskType.REGRESSION:
            label_mapper = RegressionTarget.fit(y_raw.values)
            y = label_mapper.transform(y_raw.values)
        elif task_type == TaskType.BINARY:
            label_mapper = BinaryLabelMapper.fit(y_raw.values)
            y = label_mapper.transform(y_raw.values)
        else:
            label_mapper = MultiClassLabelMapper.fit(y_raw.values)
            y = label_mapper.transform(y_raw.values)
    else:
        if task_type is None:
            task_type = TaskType.BINARY

    return Dataset(
        X=X,
        y=y,
        keys=keys,
        feature_names=feature_names,
        task_type=task_type,
        label_mapper=label_mapper,
    )


def load_tabular_parquet(
    path: Union[str, Path],
    label_col: str = "label",
    id_col: Optional[str] = "Key",
    task_type: Optional[TaskType] = None,
) -> Dataset:
    """Load a Parquet file into a Dataset.

    Args:
        path: Path to the Parquet file.
        label_col: Name of the label/target column.
        id_col: Name of the sample ID column.
        task_type: Override task type detection.

    Returns:
        Dataset with features, labels, keys, and feature names.
    """
    path = Path(path)
    df = pd.read_parquet(path)
    logger.debug("Loaded Parquet %s: %d rows × %d cols", path, len(df), len(df.columns))

    # Write to temp CSV-like flow by reusing the same logic
    if id_col is not None and id_col in df.columns:
        keys = df[id_col].astype(str).tolist()
        df = df.drop(columns=[id_col])
    else:
        keys = [f"S{i:04d}" for i in range(len(df))]

    y_raw = None
    label_mapper = None
    if label_col in df.columns:
        y_raw = df[label_col]
        df = df.drop(columns=[label_col])

    Xdf = df.select_dtypes(include=[np.number])
    feature_names = list(Xdf.columns)
    X = Xdf.to_numpy(dtype=np.float64, copy=True)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = None
    if y_raw is not None:
        if task_type is None:
            task_type = _infer_task_type(y_raw)
        if task_type == TaskType.REGRESSION:
            label_mapper = RegressionTarget.fit(y_raw.values)
            y = label_mapper.transform(y_raw.values)
        elif task_type == TaskType.BINARY:
            label_mapper = BinaryLabelMapper.fit(y_raw.values)
            y = label_mapper.transform(y_raw.values)
        else:
            label_mapper = MultiClassLabelMapper.fit(y_raw.values)
            y = label_mapper.transform(y_raw.values)
    else:
        if task_type is None:
            task_type = TaskType.BINARY

    return Dataset(
        X=X,
        y=y,
        keys=keys,
        feature_names=feature_names,
        task_type=task_type,
        label_mapper=label_mapper,
    )


def _infer_task_type(y: pd.Series) -> TaskType:
    """Infer task type from label values.

    Args:
        y: Label series.

    Returns:
        Inferred TaskType.
    """
    # Try numeric first
    y_num = pd.to_numeric(y, errors="coerce")

    # If most values are non-numeric → classification
    if y_num.isna().sum() > len(y) * 0.5:
        n_unique = y.nunique()
        return TaskType.BINARY if n_unique == 2 else TaskType.MULTICLASS

    # If numeric, check if integer-like with few classes → classification
    y_clean = y_num.dropna()
    is_int_like = all(
        float(v).is_integer() for v in y_clean
    )
    if is_int_like:
        n_unique = int(y_num.nunique())
        if n_unique == 2:
            return TaskType.BINARY
        if n_unique <= 20:
            return TaskType.MULTICLASS

    return TaskType.REGRESSION
