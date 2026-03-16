from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .encoding import amplitude_encode_matrix, AmplitudeEncodingInfo

def _is_nifti(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")

def _strip_ext(name: str) -> str:
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return Path(name).stem

def _case_id(p: Path) -> str:
    stem = _strip_ext(p.name)
    toks = [t for t in stem.split("_") if t]
    if len(toks) >= 2 and toks[-1].isdigit():
        toks = toks[:-1]
    drop = {"pet", "mask", "roi", "seg", "segmentation"}
    toks = [t for t in toks if t.lower() not in drop]
    return "_".join(toks) if toks else stem

def _parse_label(pet_path: Path) -> str:
    stem = _strip_ext(pet_path.name)
    toks = stem.split("_")
    for t in reversed(toks):
        if t.isdigit():
            return t
    raise ValueError(f"Cannot parse label from filename: {pet_path.name}")

def _read_nifti_array(path: Path) -> np.ndarray:
    import nibabel as nib
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(), dtype=np.float64)

def _collect_pairs(root: Path, pet_pattern: str, mask_pattern: Optional[str]) -> List[Tuple[Path, Optional[Path]]]:
    root = Path(root)
    pets = sorted([p for p in root.rglob("*") if p.is_file() and _is_nifti(p) and p.match(pet_pattern)])
    if len(pets) == 0:
        pets = sorted([p for p in root.rglob("*") if p.is_file() and _is_nifti(p) and ("mask" not in p.name.lower())])

    masks: Dict[str, Path] = {}
    if mask_pattern is not None:
        ms = sorted([p for p in root.rglob("*") if p.is_file() and _is_nifti(p) and p.match(mask_pattern)])
        for m in ms:
            masks[_case_id(m)] = m

    pairs: List[Tuple[Path, Optional[Path]]] = []
    for pet in pets:
        pairs.append((pet, masks.get(_case_id(pet), None)))
    return pairs

@dataclass
class NiftiDatasetSplit:
    X_raw: np.ndarray
    X_amp: np.ndarray
    y_raw: np.ndarray
    ids: List[str]

def _build_split(pairs: List[Tuple[Path, Optional[Path]]], pad_len: Optional[int]) -> Tuple[NiftiDatasetSplit, AmplitudeEncodingInfo]:
    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    ids: List[str] = []

    for pet_path, mask_path in pairs:
        img = _read_nifti_array(pet_path)
        if mask_path is not None:
            m = _read_nifti_array(mask_path)
            if img.shape != m.shape:
                raise ValueError(f"Shape mismatch {pet_path.name} vs {mask_path.name}: {img.shape} vs {m.shape}")
            img = np.where(m > 0, img, 0.0)
        vec = img.ravel().astype(np.float64)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        X_list.append(vec)
        y_list.append(_parse_label(pet_path))
        ids.append(_case_id(pet_path))

    max_len = max(v.size for v in X_list) if X_list else 0
    X_raw = np.zeros((len(X_list), max_len), dtype=np.float64)
    for i, v in enumerate(X_list):
        X_raw[i, :v.size] = v

    X_amp, info = amplitude_encode_matrix(X_raw, pad_len=pad_len)
    return NiftiDatasetSplit(X_raw=X_raw, X_amp=X_amp, y_raw=np.array(y_list), ids=ids), info

def load_nifti_dataset(
    root: Path,
    pet_pattern: str = "*PET*.nii*",
    mask_pattern: Optional[str] = "*mask*.nii*",
    pad_len: Optional[int] = None,
) -> Dict[str, Any]:
    """Load NIfTI dataset.

    If Train/ and Test/ exist, returns explicit splits (both raw and amplitude).
    Otherwise returns a single set; caller splits.
    """
    root = Path(root)
    train_dir = root / "Train"
    test_dir = root / "Test"

    if train_dir.exists() and test_dir.exists():
        tr_pairs = _collect_pairs(train_dir, pet_pattern, mask_pattern)
        te_pairs = _collect_pairs(test_dir, pet_pattern, mask_pattern)

        tr, info_tr = _build_split(tr_pairs, pad_len=pad_len)
        te, _ = _build_split(te_pairs, pad_len=info_tr.pad_len)

        return {
            "name": root.name,
            "train": tr,
            "test": te,
            "amp_info": info_tr,
            "preprocess": {"pet_pattern": pet_pattern, "mask_pattern": mask_pattern},
        }

    pairs = _collect_pairs(root, pet_pattern, mask_pattern)
    all_split, info = _build_split(pairs, pad_len=pad_len)
    return {
        "name": root.name,
        "all": all_split,
        "amp_info": info,
        "preprocess": {"pet_pattern": pet_pattern, "mask_pattern": mask_pattern},
    }

def load_nifti_for_inference(
    root: Path,
    pet_pattern: str,
    mask_pattern: Optional[str],
    pad_len: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], AmplitudeEncodingInfo]:
    pairs = _collect_pairs(Path(root), pet_pattern, mask_pattern)
    split, info = _build_split(pairs, pad_len=pad_len)
    return split.X_raw, split.X_amp, split.ids, info
