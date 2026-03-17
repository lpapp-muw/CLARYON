"""Binary grouping — user-defined binary reduction of multiclass labels."""
from __future__ import annotations

import logging

import numpy as np

from ..config_schema import BinaryGroupingConfig

logger = logging.getLogger(__name__)


def apply_binary_grouping(
    y: np.ndarray,
    config: BinaryGroupingConfig,
) -> np.ndarray:
    """Apply binary grouping to multiclass labels.

    Args:
        y: Original label array (integer-encoded).
        config: Binary grouping configuration.

    Returns:
        Binary label array (0/1) if enabled, or original labels if disabled.
    """
    if not config.enabled:
        return y

    positive_set = set(config.positive)
    negative_set = set(config.negative) if config.negative else None

    y_new = np.zeros(len(y), dtype=np.int64)
    for i, val in enumerate(y):
        v = int(val) if hasattr(val, 'item') else val
        if v in positive_set:
            y_new[i] = 1
        elif negative_set is not None:
            y_new[i] = 0 if v in negative_set else 0
        else:
            y_new[i] = 0

    logger.info(
        "Binary grouping applied: positive=%s, negative=%s → %d pos / %d neg",
        config.positive,
        config.negative or "everything else",
        int(y_new.sum()),
        int((y_new == 0).sum()),
    )
    return y_new
