"""Core data containers: Dataset dataclass and LabelMapper hierarchy."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Supervised learning task types."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


def _to_py_scalar(v: Any) -> Any:
    """Convert numpy scalars to native Python types."""
    if isinstance(v, np.generic):
        return v.item()
    return v


# ═══════════════════════════════════════════════════════════════
# Label mappers
# ═══════════════════════════════════════════════════════════════


@dataclass
class BinaryLabelMapper:
    """Maps two-class labels to 0/1 integers and back.

    Attributes:
        classes: Sorted list of the two original label values.
        to_int: Mapping from original label → 0 or 1.
        to_label: Mapping from 0/1 → original label.
    """

    classes: List[Any]
    to_int: Dict[Any, int]
    to_label: Dict[int, Any]

    @staticmethod
    def fit(y: Sequence[Any]) -> BinaryLabelMapper:
        """Fit mapper from a label sequence.

        Args:
            y: Sequence of label values (must contain exactly 2 unique values).

        Returns:
            Fitted BinaryLabelMapper.

        Raises:
            ValueError: If y does not contain exactly 2 unique values.
        """
        uniq = pd.unique(pd.Series(y))
        if len(uniq) != 2:
            raise ValueError(
                f"Binary classification requires exactly 2 unique labels, got {len(uniq)}: {list(uniq)}"
            )
        classes = sorted([_to_py_scalar(v) for v in uniq], key=lambda x: str(x))
        to_int = {classes[0]: 0, classes[1]: 1}
        to_label = {0: classes[0], 1: classes[1]}
        return BinaryLabelMapper(classes=classes, to_int=to_int, to_label=to_label)

    def transform(self, y: Sequence[Any]) -> np.ndarray:
        """Map original labels to 0/1 integers.

        Args:
            y: Sequence of original label values.

        Returns:
            Integer array of shape (len(y),).
        """
        return np.array([self.to_int[_to_py_scalar(v)] for v in y], dtype=np.int64)

    def inverse_transform(self, y_int: np.ndarray) -> List[Any]:
        """Map 0/1 integers back to original labels.

        Args:
            y_int: Integer array.

        Returns:
            List of original label values.
        """
        return [self.to_label[int(v)] for v in y_int.tolist()]

    def to_json(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {"type": "binary", "classes": [_to_py_scalar(c) for c in self.classes]}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> BinaryLabelMapper:
        """Deserialize from JSON dict."""
        classes = [d["classes"][0], d["classes"][1]]
        to_int = {classes[0]: 0, classes[1]: 1}
        to_label = {0: classes[0], 1: classes[1]}
        return BinaryLabelMapper(classes=classes, to_int=to_int, to_label=to_label)


@dataclass
class MultiClassLabelMapper:
    """Maps K-class labels to 0..K-1 integers and back.

    Attributes:
        classes: Sorted list of original label values.
        to_int: Mapping from original label → integer.
        to_label: Mapping from integer → original label.
    """

    classes: List[Any]
    to_int: Dict[Any, int]
    to_label: Dict[int, Any]

    @staticmethod
    def fit(y: Sequence[Any]) -> MultiClassLabelMapper:
        """Fit mapper from a label sequence.

        Args:
            y: Sequence of label values (must contain ≥2 unique values).

        Returns:
            Fitted MultiClassLabelMapper.

        Raises:
            ValueError: If y contains fewer than 2 unique values.
        """
        uniq = pd.unique(pd.Series(y))
        if len(uniq) < 2:
            raise ValueError(f"Classification requires ≥2 unique labels, got {len(uniq)}")
        classes = sorted([_to_py_scalar(v) for v in uniq], key=lambda x: str(x))
        to_int = {c: i for i, c in enumerate(classes)}
        to_label = {i: c for i, c in enumerate(classes)}
        return MultiClassLabelMapper(classes=classes, to_int=to_int, to_label=to_label)

    def transform(self, y: Sequence[Any]) -> np.ndarray:
        """Map original labels to 0..K-1 integers.

        Args:
            y: Sequence of original label values.

        Returns:
            Integer array of shape (len(y),).
        """
        return np.array([self.to_int[_to_py_scalar(v)] for v in y], dtype=np.int64)

    def inverse_transform(self, y_int: np.ndarray) -> List[Any]:
        """Map 0..K-1 integers back to original labels.

        Args:
            y_int: Integer array.

        Returns:
            List of original label values.
        """
        return [self.to_label[int(v)] for v in y_int.tolist()]

    @property
    def n_classes(self) -> int:
        """Number of classes."""
        return len(self.classes)

    def to_json(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {"type": "multiclass", "classes": [_to_py_scalar(c) for c in self.classes]}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> MultiClassLabelMapper:
        """Deserialize from JSON dict."""
        classes = list(d["classes"])
        to_int = {c: i for i, c in enumerate(classes)}
        to_label = {i: c for i, c in enumerate(classes)}
        return MultiClassLabelMapper(classes=classes, to_int=to_int, to_label=to_label)


@dataclass
class RegressionTarget:
    """Pass-through container for regression targets (no mapping needed).

    Stores optional statistics for denormalization.
    """

    mean: Optional[float] = None
    std: Optional[float] = None

    @staticmethod
    def fit(y: Sequence[float]) -> RegressionTarget:
        """Compute target statistics.

        Args:
            y: Sequence of continuous target values.

        Returns:
            Fitted RegressionTarget with mean and std.
        """
        arr = np.asarray(y, dtype=np.float64)
        return RegressionTarget(mean=float(arr.mean()), std=float(arr.std()))

    def transform(self, y: Sequence[float]) -> np.ndarray:
        """Convert to float64 array (no mapping).

        Args:
            y: Sequence of target values.

        Returns:
            Float64 array of shape (len(y),).
        """
        return np.asarray(y, dtype=np.float64)

    def inverse_transform(self, y: np.ndarray) -> List[float]:
        """Convert array back to list of floats.

        Args:
            y: Float array.

        Returns:
            List of float values.
        """
        return y.astype(np.float64).tolist()

    def to_json(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {"type": "regression", "mean": self.mean, "std": self.std}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> RegressionTarget:
        """Deserialize from JSON dict."""
        return RegressionTarget(mean=d.get("mean"), std=d.get("std"))


# ═══════════════════════════════════════════════════════════════
# Dataset container
# ═══════════════════════════════════════════════════════════════


@dataclass
class Dataset:
    """Unified dataset container for all modalities.

    Attributes:
        X: Feature matrix, shape (n_samples, n_features) or image array.
        y: Target array, shape (n_samples,). None for inference-only data.
        keys: Sample identifiers (patient/subject IDs).
        feature_names: Column names for tabular data.
        task_type: Learning task type.
        label_mapper: Maps raw labels ↔ integers (classification) or holds
            target stats (regression).
        data_source: Origin of the data — "tabular" (CSV), "imaging" (flattened
            NIfTI/TIFF), or "fused" (concatenated tabular + imaging).
        metadata: Arbitrary extra metadata.
    """

    X: np.ndarray
    y: Optional[np.ndarray] = None
    keys: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    task_type: TaskType = TaskType.BINARY
    label_mapper: Any = None
    data_source: str = "tabular"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features (for 2D tabular data)."""
        if self.X.ndim < 2:
            return 1
        return self.X.shape[1]
