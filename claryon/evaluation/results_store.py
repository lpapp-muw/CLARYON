"""Results table builder — aggregates predictions into summary tables.

Rewritten from [B] results_collector.py. Consumes unified Predictions.csv format.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..io.predictions import read_predictions, infer_task_type, SEP
from ..io.base import TaskType

logger = logging.getLogger(__name__)


def collect_results(
    results_dir: Path,
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Collect all Predictions.csv files into a summary DataFrame.

    Args:
        results_dir: Root results directory.
        models: Filter to specific model names. If None, collect all.

    Returns:
        DataFrame with columns: model, seed, fold, key, actual, predicted, + probability cols.
    """
    results_dir = Path(results_dir)
    rows = []

    for pred_path in sorted(results_dir.rglob("Predictions.csv")):
        parts = pred_path.relative_to(results_dir).parts
        if len(parts) < 4:
            continue

        model_name = parts[0]
        if models is not None and model_name not in models:
            continue

        seed_str = parts[1].replace("seed_", "")
        fold_str = parts[2].replace("fold_", "")

        try:
            df = read_predictions(pred_path)
            df["model"] = model_name
            df["seed"] = int(seed_str)
            df["fold"] = int(fold_str)
            rows.append(df)
        except Exception as e:
            logger.warning("Failed to read %s: %s", pred_path, e)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def compute_fold_metrics(
    results_df: pd.DataFrame,
    metric_fns: Dict[str, Any],
) -> pd.DataFrame:
    """Compute metrics per model/seed/fold.

    Args:
        results_df: DataFrame from collect_results().
        metric_fns: Dict of metric_name → callable(y_true, y_pred, probabilities=...).

    Returns:
        DataFrame with columns: model, seed, fold, + metric columns.
    """
    rows = []
    grouped = results_df.groupby(["model", "seed", "fold"])

    for (model, seed, fold), group in grouped:
        y_true = group["Actual"].values
        y_pred = group["Predicted"].values

        prob_cols = sorted(
            [c for c in group.columns if c.startswith("P") and c[1:].isdigit()],
            key=lambda c: int(c[1:]),
        )
        probs = group[prob_cols].values if prob_cols else None

        row: Dict[str, Any] = {"model": model, "seed": int(seed), "fold": int(fold)}
        for name, fn in metric_fns.items():
            try:
                row[name] = fn(y_true, y_pred, probabilities=probs)
            except Exception as e:
                row[name] = float("nan")
                logger.debug("Metric %s failed for %s/%s/%s: %s", name, model, seed, fold, e)

        rows.append(row)

    return pd.DataFrame(rows)
