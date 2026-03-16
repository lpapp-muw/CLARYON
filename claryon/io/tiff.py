"""TIFF image loader — TIFF files + metadata sidecar → Dataset.

New module. Loads multi-page TIFF images with optional JSON metadata sidecar.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BinaryLabelMapper, Dataset, MultiClassLabelMapper, TaskType

logger = logging.getLogger(__name__)


def load_tiff_dataset(
    root: Union[str, Path],
    metadata_file: Optional[str] = "metadata.json",
    task_type: Optional[TaskType] = None,
) -> Dataset:
    """Load TIFF images from a directory into a Dataset.

    Expects a directory with .tif/.tiff files and an optional metadata JSON
    sidecar containing labels and sample IDs.

    Args:
        root: Directory containing TIFF files.
        metadata_file: Name of JSON sidecar file with labels/IDs.
            Expected format: ``{"samples": [{"file": "a.tif", "label": 0, "id": "S0"}, ...]}``
        task_type: Override task type. If None, inferred from labels.

    Returns:
        Dataset with flattened image features.
    """
    root = Path(root)
    tiff_files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
    )

    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {root}")

    # Load metadata: either single metadata.json or per-file JSON sidecars
    samples_meta: Dict[str, Dict[str, Any]] = {}
    meta_path = root / metadata_file if metadata_file else None
    if meta_path and meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        samples_meta = {s["file"]: s for s in metadata.get("samples", [])}

    X_list: List[np.ndarray] = []
    y_list: List[Any] = []
    keys: List[str] = []
    has_labels = False

    for tf in tiff_files:
        img = _read_tiff(tf)
        vec = img.ravel().astype(np.float64)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        X_list.append(vec)

        # Try per-file JSON sidecar if not in single metadata.json
        sm = samples_meta.get(tf.name, {})
        if not sm:
            sidecar_path = tf.with_suffix(".json")
            if sidecar_path.exists():
                with open(sidecar_path) as f:
                    sm = json.load(f)

        keys.append(sm.get("id", tf.stem))
        if "label" in sm:
            y_list.append(sm["label"])
            has_labels = True
        else:
            y_list.append(None)

    # Pad to uniform length
    max_len = max(v.size for v in X_list)
    X = np.zeros((len(X_list), max_len), dtype=np.float64)
    for i, v in enumerate(X_list):
        X[i, : v.size] = v

    y = None
    label_mapper = None
    if has_labels and all(v is not None for v in y_list):
        if task_type is None:
            n_unique = len(set(y_list))
            task_type = TaskType.BINARY if n_unique == 2 else TaskType.MULTICLASS

        if task_type == TaskType.BINARY:
            label_mapper = BinaryLabelMapper.fit(y_list)
        elif task_type == TaskType.MULTICLASS:
            label_mapper = MultiClassLabelMapper.fit(y_list)

        if label_mapper is not None:
            y = label_mapper.transform(y_list)
    else:
        if task_type is None:
            task_type = TaskType.BINARY

    return Dataset(
        X=X,
        y=y,
        keys=keys,
        task_type=task_type,
        label_mapper=label_mapper,
    )


def _read_tiff(path: Path) -> np.ndarray:
    """Read a TIFF file as a numpy array.

    Args:
        path: Path to .tif/.tiff file.

    Returns:
        Image array.
    """
    import tifffile

    return tifffile.imread(str(path))
