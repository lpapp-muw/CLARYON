"""Unified Predictions.csv writer/reader (REQ §8.4).

All model outputs are serialized through this module.
Models never write CSVs directly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base import TaskType

logger = logging.getLogger(__name__)

SEP = ";"
FLOAT_FMT = "%.8f"


def write_predictions(
    path: Union[str, Path],
    keys: List[str],
    actual: Optional[np.ndarray],
    predicted: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    task_type: TaskType = TaskType.BINARY,
    threshold: Optional[float] = None,
    fold: Optional[int] = None,
    seed: Optional[int] = None,
) -> Path:
    """Write predictions to a semicolon-separated CSV file.

    Args:
        path: Output file path.
        keys: Sample identifiers, length N.
        actual: Ground truth values, shape (N,). None for inference-only.
        predicted: Predicted values, shape (N,). Class labels for classification,
            continuous values for regression.
        probabilities: Per-class probabilities, shape (N, K). Required for
            classification tasks. None for regression.
        task_type: Learning task type.
        threshold: Decision threshold (binary classification only).
        fold: CV fold index (optional metadata column).
        seed: CV seed (optional metadata column).

    Returns:
        The path written to.

    Raises:
        ValueError: If shapes are inconsistent or required data is missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(keys)
    if predicted.shape[0] != n:
        raise ValueError(f"keys has {n} entries but predicted has {predicted.shape[0]}")

    rows: Dict[str, list] = {"Key": keys}

    # Actual column
    if actual is not None:
        if actual.shape[0] != n:
            raise ValueError(f"keys has {n} entries but actual has {actual.shape[0]}")
        rows["Actual"] = actual.tolist()
    else:
        rows["Actual"] = [""] * n

    # Predicted column
    rows["Predicted"] = predicted.tolist()

    # Probability columns (classification only)
    if task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        if probabilities is None:
            raise ValueError("probabilities required for classification tasks")
        if probabilities.shape[0] != n:
            raise ValueError(f"keys has {n} entries but probabilities has {probabilities.shape[0]}")
        n_classes = probabilities.shape[1]
        for k in range(n_classes):
            rows[f"P{k}"] = probabilities[:, k].tolist()

    # Optional metadata columns
    if threshold is not None:
        rows["Threshold"] = [threshold] * n
    if fold is not None:
        rows["Fold"] = [fold] * n
    if seed is not None:
        rows["Seed"] = [seed] * n

    df = pd.DataFrame(rows)

    # Format floats
    float_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
    if task_type == TaskType.REGRESSION:
        float_cols.extend(["Predicted"])
        if actual is not None:
            float_cols.append("Actual")
    if "Threshold" in df.columns:
        float_cols.append("Threshold")

    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda v: f"{v:.8f}" if isinstance(v, float) else v)

    df.to_csv(path, sep=SEP, index=False)
    logger.debug("Wrote predictions to %s (%d rows)", path, n)
    return path


def read_predictions(path: Union[str, Path]) -> pd.DataFrame:
    """Read a Predictions.csv file.

    Args:
        path: Path to the predictions CSV.

    Returns:
        DataFrame with columns as written by write_predictions().

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    df = pd.read_csv(path, sep=SEP)
    logger.debug("Read predictions from %s (%d rows)", path, len(df))
    return df


def infer_task_type(df: pd.DataFrame) -> TaskType:
    """Infer the task type from a predictions DataFrame.

    Args:
        df: DataFrame read by read_predictions().

    Returns:
        Inferred TaskType.
    """
    prob_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
    if len(prob_cols) == 0:
        return TaskType.REGRESSION
    elif len(prob_cols) == 2:
        return TaskType.BINARY
    else:
        return TaskType.MULTICLASS
