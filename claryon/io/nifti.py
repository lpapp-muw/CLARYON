"""NIfTI medical image loader — volumes + masks → Dataset.

Ported from [E] nifti.py. Encoding decoupled: raw voxel data only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BinaryLabelMapper, Dataset, MultiClassLabelMapper, TaskType

logger = logging.getLogger(__name__)


def _is_nifti(p: Path) -> bool:
    """Check if a path is a NIfTI file."""
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def _strip_ext(name: str) -> str:
    """Remove NIfTI extensions from a filename."""
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def _case_id(p: Path) -> str:
    """Extract case ID from filename, dropping modality/mask suffixes.

    Args:
        p: Path to a NIfTI file.

    Returns:
        Case identifier string.
    """
    stem = _strip_ext(p.name)
    toks = [t for t in stem.split("_") if t]
    # Drop trailing digit token (label encoded in filename)
    if len(toks) >= 2 and toks[-1].isdigit():
        toks = toks[:-1]
    drop = {"pet", "mask", "roi", "seg", "segmentation"}
    toks = [t for t in toks if t.lower() not in drop]
    return "_".join(toks) if toks else stem


def _parse_label(pet_path: Path) -> str:
    """Extract numeric label from trailing digits in filename.

    Args:
        pet_path: Path to PET/image NIfTI file.

    Returns:
        Label string.

    Raises:
        ValueError: If no numeric label found in filename.
    """
    stem = _strip_ext(pet_path.name)
    toks = stem.split("_")
    for t in reversed(toks):
        if t.isdigit():
            return t
    raise ValueError(f"Cannot parse label from filename: {pet_path.name}")


def _read_nifti_array(path: Path) -> np.ndarray:
    """Load a NIfTI file as a float64 numpy array.

    Args:
        path: Path to .nii or .nii.gz file.

    Returns:
        Array of voxel values.
    """
    import nibabel as nib

    img = nib.load(str(path))
    return np.asarray(img.get_fdata(), dtype=np.float64)


def _collect_pairs(
    root: Path,
    pet_pattern: str,
    mask_pattern: Optional[str],
) -> List[Tuple[Path, Optional[Path]]]:
    """Match image volumes to their masks within a directory.

    Args:
        root: Directory to search.
        pet_pattern: Glob pattern for image volumes.
        mask_pattern: Glob pattern for mask volumes. None = no masking.

    Returns:
        List of (image_path, mask_path_or_None) tuples.
    """
    root = Path(root)
    pets = sorted(
        p for p in root.rglob("*")
        if p.is_file() and _is_nifti(p) and p.match(pet_pattern)
    )
    if not pets:
        # Fallback: all NIfTI files that aren't masks
        pets = sorted(
            p for p in root.rglob("*")
            if p.is_file() and _is_nifti(p) and "mask" not in p.name.lower()
        )

    masks: Dict[str, Path] = {}
    if mask_pattern is not None:
        ms = sorted(
            p for p in root.rglob("*")
            if p.is_file() and _is_nifti(p) and p.match(mask_pattern)
        )
        for m in ms:
            masks[_case_id(m)] = m

    pairs: List[Tuple[Path, Optional[Path]]] = []
    for pet in pets:
        pairs.append((pet, masks.get(_case_id(pet))))
    return pairs


def _build_arrays(
    pairs: List[Tuple[Path, Optional[Path]]],
    flatten_order: str = "rowmajor",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load and flatten NIfTI pairs into feature matrix.

    Args:
        pairs: List of (image_path, mask_path_or_None).

    Returns:
        (X, y_labels, ids, raw_shapes) — X is (N, max_voxels), y_labels is string
        array of parsed labels, ids are case IDs.
    """
    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    ids: List[str] = []

    for pet_path, mask_path in pairs:
        img = _read_nifti_array(pet_path)
        if mask_path is not None:
            m = _read_nifti_array(mask_path)
            if img.shape != m.shape:
                raise ValueError(
                    f"Shape mismatch {pet_path.name} vs {mask_path.name}: "
                    f"{img.shape} vs {m.shape}"
                )
            img = np.where(m > 0, img, 0.0)

        from .hilbert import flatten_volume

        vec = flatten_volume(img, order=flatten_order).astype(np.float64)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        X_list.append(vec)
        y_list.append(_parse_label(pet_path))
        ids.append(_case_id(pet_path))

    # Pad to uniform length (determined by largest volume in cohort)
    max_len = max(v.size for v in X_list) if X_list else 0
    sizes = [v.size for v in X_list]
    min_len = min(sizes) if sizes else 0

    if min_len < max_len:
        import math
        n_padded = sum(1 for s in sizes if s < max_len)
        logger.warning(
            "NIfTI volumes have different flattened lengths (min=%d, max=%d). "
            "%d/%d volumes will be zero-padded to %d. For best results, ensure "
            "all volumes in a cohort have identical dimensions.",
            min_len, max_len, n_padded, len(sizes), max_len,
        )

    # Log qubit requirements for quantum models
    import math
    next_pow2 = 1 << int(math.ceil(math.log2(max(max_len, 1))))
    n_qubits = int(math.ceil(math.log2(max(max_len, 1))))
    logger.info(
        "NIfTI cohort: %d samples, %d voxels per sample (flattened). "
        "Amplitude encoding will require %d qubits (padded to %d).",
        len(X_list), max_len, n_qubits, next_pow2,
    )

    X = np.zeros((len(X_list), max_len), dtype=np.float64)
    for i, v in enumerate(X_list):
        X[i, : v.size] = v

    return X, np.array(y_list), ids, []


def load_nifti_dataset(
    root: Union[str, Path],
    pet_pattern: str = "*.nii*",
    mask_pattern: Optional[str] = "*mask*.nii*",
    task_type: Optional[TaskType] = None,
    flatten_order: str = "rowmajor",
) -> Dict[str, Any]:
    """Load a NIfTI dataset into Dataset objects.

    If Train/ and Test/ subdirectories exist, returns a dict with ``"train"``
    and ``"test"`` Dataset entries. Otherwise returns ``"all"`` as a single
    Dataset.

    Args:
        root: Root directory containing NIfTI files.
        pet_pattern: Glob pattern for image volumes.
        mask_pattern: Glob pattern for mask volumes. None disables masking.
        task_type: Override task type. If None, inferred from labels.

    Returns:
        Dict with ``"name"``, ``"train"``/``"test"`` or ``"all"`` Dataset(s),
        and ``"metadata"`` dict.
    """
    root = Path(root)
    train_dir = root / "Train"
    test_dir = root / "Test"

    if train_dir.exists() and test_dir.exists():
        tr_pairs = _collect_pairs(train_dir, pet_pattern, mask_pattern)
        te_pairs = _collect_pairs(test_dir, pet_pattern, mask_pattern)

        X_tr, y_tr, ids_tr, _ = _build_arrays(tr_pairs, flatten_order=flatten_order)
        X_te, y_te, ids_te, _ = _build_arrays(te_pairs, flatten_order=flatten_order)

        if task_type is None:
            n_unique = len(set(y_tr.tolist()))
            task_type = TaskType.BINARY if n_unique == 2 else TaskType.MULTICLASS

        label_mapper = _fit_label_mapper(y_tr, task_type)
        y_tr_int = label_mapper.transform(y_tr) if label_mapper else None
        y_te_int = label_mapper.transform(y_te) if label_mapper else None

        return {
            "name": root.name,
            "train": Dataset(
                X=X_tr, y=y_tr_int, keys=ids_tr,
                task_type=task_type, label_mapper=label_mapper,
            ),
            "test": Dataset(
                X=X_te, y=y_te_int, keys=ids_te,
                task_type=task_type, label_mapper=label_mapper,
            ),
            "metadata": {"pet_pattern": pet_pattern, "mask_pattern": mask_pattern},
        }

    # Single directory
    pairs = _collect_pairs(root, pet_pattern, mask_pattern)
    X, y_labels, ids, _ = _build_arrays(pairs, flatten_order=flatten_order)

    if task_type is None:
        n_unique = len(set(y_labels.tolist()))
        task_type = TaskType.BINARY if n_unique == 2 else TaskType.MULTICLASS

    label_mapper = _fit_label_mapper(y_labels, task_type)
    y_int = label_mapper.transform(y_labels) if label_mapper else None

    return {
        "name": root.name,
        "all": Dataset(
            X=X, y=y_int, keys=ids,
            task_type=task_type, label_mapper=label_mapper,
        ),
        "metadata": {"pet_pattern": pet_pattern, "mask_pattern": mask_pattern},
    }


def _fit_label_mapper(
    y_labels: np.ndarray,
    task_type: TaskType,
) -> Any:
    """Create appropriate label mapper for NIfTI labels.

    Args:
        y_labels: String array of label values.
        task_type: Task type to use.

    Returns:
        BinaryLabelMapper or MultiClassLabelMapper.
    """
    if task_type == TaskType.BINARY:
        return BinaryLabelMapper.fit(y_labels.tolist())
    return MultiClassLabelMapper.fit(y_labels.tolist())
