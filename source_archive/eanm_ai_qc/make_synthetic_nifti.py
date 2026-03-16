#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import nibabel as nib

def write_nii_gz(path: Path, arr: np.ndarray) -> None:
    arr = arr.astype(np.float32)
    affine = np.eye(4, dtype=np.float32)
    img = nib.Nifti1Image(arr, affine)
    nib.save(img, str(path))

def make_case(shape, label, masked: bool):
    rng = np.random.default_rng(int(label) + 123)
    pet = rng.normal(loc=0.0, scale=0.2, size=shape).astype(np.float32)
    z0, y0, x0 = [s // 4 for s in shape]
    z1, y1, x1 = [s - s // 4 for s in shape]
    roi = np.zeros(shape, dtype=np.float32)
    roi[z0:z1, y0:y1, x0:x1] = 1.0
    if int(label) == 1:
        pet += roi * 2.0
    if masked:
        return pet, roi.copy()
    return pet, None

def build_dataset(root: Path, with_masks: bool, shape=(10, 12, 8), n_train=10, n_test=6):
    (root / "Train").mkdir(parents=True, exist_ok=True)
    (root / "Test").mkdir(parents=True, exist_ok=True)
    for split, n in [("Train", n_train), ("Test", n_test)]:
        for i in range(n):
            label = i % 2
            pet, mask = make_case(shape, label, masked=with_masks)
            case = f"case{split.lower()}{i:03d}_{label}"
            pet_path = root / split / f"{case}_PET.nii.gz"
            write_nii_gz(pet_path, pet)
            if with_masks:
                mask_path = root / split / f"{case}_mask.nii.gz"
                write_nii_gz(mask_path, mask)

base = Path(__file__).resolve().parents[1] / "demo_data"
masked_root = base / "nifti_masked"
nomask_root = base / "nifti_nomask"

build_dataset(masked_root, with_masks=True)
build_dataset(nomask_root, with_masks=False)

print(f"Wrote masked PET+mask dataset: {masked_root}")
print(f"Wrote PET-only dataset:        {nomask_root}")
