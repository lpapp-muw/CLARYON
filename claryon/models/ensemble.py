"""Ensemble aggregation — softmax averaging for classification, mean for regression.

Ported from [B] ensemble_aggregator.py. Extended with regression support.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..io.base import TaskType

logger = logging.getLogger(__name__)


def ensemble_predictions(
    prob_arrays: List[np.ndarray],
    task_type: TaskType = TaskType.BINARY,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate predictions from multiple models via averaging.

    For classification: averages softmax probabilities, then takes argmax.
    For regression: averages predicted values.

    Args:
        prob_arrays: List of arrays to average. For classification, each is
            shape (N, C) probability matrix. For regression, each is shape (N,)
            predicted values.
        task_type: Learning task type.

    Returns:
        Tuple of (predictions, averaged_values). For classification: (class_labels, avg_probs).
        For regression: (avg_values, avg_values).

    Raises:
        ValueError: If prob_arrays is empty or shapes don't match.
    """
    if not prob_arrays:
        raise ValueError("No prediction arrays to ensemble")

    if task_type == TaskType.REGRESSION:
        stacked = np.stack(prob_arrays, axis=0)  # (K, N)
        avg = stacked.mean(axis=0)
        return avg, avg

    # Classification: average probabilities
    stacked = np.stack(prob_arrays, axis=0)  # (K, N, C)
    avg_probs = stacked.mean(axis=0)  # (N, C)
    predictions = np.argmax(avg_probs, axis=1)

    return predictions, avg_probs
