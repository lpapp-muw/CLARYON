#!/usr/bin/env python3
"""Run SHAP + LIME explanations on already-trained models saved under Results/.

This script assumes you already ran training (e.g. via examples/run_all_examples.sh).
It does NOT train models. It loads:
- Results/<dataset>/<method>/model/metadata.json
- Results/<dataset>/<method>/model/* (weights)
- Results/<dataset>/<method>/predictions/train.csv and test.csv (to reconstruct train/test splits by id)

It then runs SHAP and LIME and writes into:
- Results/<dataset>/<method>/explain/shap/
- Results/<dataset>/<method>/explain/lime/

Runtime note:
- Explanations can be slow, especially for the quantum-kernel SVM, because each model evaluation
  can imply many quantum-circuit simulations. Use the caps below to control runtime.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is on sys.path so `eanm_ai_qc` can be imported when
# running as `python examples/run_explain_all.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from eanm_ai_qc.io.tabular import load_tabular_csv
from eanm_ai_qc.io.nifti import load_nifti_dataset
from eanm_ai_qc.io.encoding import amplitude_encode_matrix
from eanm_ai_qc.explain import run_shap, run_lime
from eanm_ai_qc.models import PLAmplitudeKernelSVM, PLQCNN_MUW, PLQCNN_Alt

METHOD_LOADERS = {
    "pl_kernel_svm": PLAmplitudeKernelSVM.load,
    "pl_qcnn_muw": PLQCNN_MUW.load,
    "pl_qcnn_alt": PLQCNN_Alt.load,
}

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _resolve_input_path(meta_path: Path, input_path_str: str) -> Path:
    p = Path(input_path_str)
    if p.is_absolute():
        return p
    # treat as repo-relative (typical)
    return (_repo_root() / p).resolve()

def _read_ids(pred_csv: Path) -> List[str]:
    df = pd.read_csv(pred_csv)
    if "id" not in df.columns:
        raise ValueError(f"Missing 'id' column in {pred_csv}")
    return df["id"].astype(str).tolist()

def _subset_by_ids(ids_all: List[str], X_raw: np.ndarray, ids_keep: List[str]) -> np.ndarray:
    m = {str(i): idx for idx, i in enumerate(ids_all)}
    idxs = [m[str(i)] for i in ids_keep if str(i) in m]
    if len(idxs) == 0:
        raise ValueError("No ids matched between predictions and loaded dataset")
    return X_raw[np.array(idxs, dtype=int)]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="Results", help="Folder created by runs (default: Results)")
    ap.add_argument("--datasets", default=None, help="Comma-separated dataset names to explain (default: all in Results/)")
    ap.add_argument("--methods", default=None, help="Comma-separated method names to explain (default: all found per dataset)")
    ap.add_argument("--max-features", type=int, default=32, help="Max raw features used by SHAP/LIME (variance-ranked)")
    ap.add_argument("--max-test-samples", type=int, default=5, help="Max test cases explained (per method)")
    ap.add_argument("--background-samples", type=int, default=12, help="Background samples for SHAP")
    ap.add_argument("--lime-num-samples", type=int, default=800, help="LIME perturbed samples per explained case")
    ap.add_argument("--shap-max-evals", type=int, default=512, help="SHAP max model evaluations (PermutationExplainer)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Missing results dir: {results_dir}")

    ds_filter = None
    if args.datasets:
        ds_filter = {d.strip() for d in args.datasets.split(",") if d.strip()}

    method_filter = None
    if args.methods:
        method_filter = {m.strip() for m in args.methods.split(",") if m.strip()}

    datasets = [p for p in sorted(results_dir.iterdir()) if p.is_dir()]
    if ds_filter is not None:
        datasets = [p for p in datasets if p.name in ds_filter]

    if not datasets:
        raise SystemExit("No datasets found to explain.")

    for ds_dir in datasets:
        print(f"[EXPLAIN] dataset={ds_dir.name}")
        method_dirs = [p for p in sorted(ds_dir.iterdir()) if p.is_dir()]
        # method folders are those that contain model/metadata.json
        method_dirs = [p for p in method_dirs if (p/"model"/"metadata.json").exists()]
        if method_filter is not None:
            method_dirs = [p for p in method_dirs if p.name in method_filter]

        if not method_dirs:
            print(f"[EXPLAIN] dataset={ds_dir.name} no method folders found; skipping")
            continue

        for mdir in method_dirs:
            method = mdir.name
            meta_path = mdir / "model" / "metadata.json"
            meta = pd.read_json(meta_path, typ="series").to_dict()
            input_type = str(meta.get("input_type", "tabular"))
            input_path = _resolve_input_path(meta_path, str(meta.get("input_path", "")))
            pad_len = int(meta.get("pad_len"))

            if method not in METHOD_LOADERS:
                print(f"[EXPLAIN] dataset={ds_dir.name} method={method} unknown loader; skipping")
                continue

            # Load model
            model = METHOD_LOADERS[method](mdir / "model")

            # Reconstruct train/test raw matrices from saved prediction ids
            train_pred = mdir / "predictions" / "train.csv"
            test_pred = mdir / "predictions" / "test.csv"
            if not train_pred.exists() or not test_pred.exists():
                print(f"[EXPLAIN] dataset={ds_dir.name} method={method} missing train/test prediction CSVs; skipping")
                continue

            train_ids = _read_ids(train_pred)
            test_ids = _read_ids(test_pred)

            # Load dataset raw features (full), then subset
            if input_type == "tabular":
                ds = load_tabular_csv(input_path, label_col="label")
                ids_all = ds.ids
                X_raw_all = ds.X_raw
                feature_names_raw = ds.feature_names
                X_train_raw = _subset_by_ids(ids_all, X_raw_all, train_ids)
                X_test_raw = _subset_by_ids(ids_all, X_raw_all, test_ids)

            elif input_type == "nifti":
                loaded = load_nifti_dataset(input_path)
                # For explicit Train/Test datasets, ids are split across the two.
                if "train" in loaded:
                    tr = loaded["train"]
                    te = loaded["test"]
                    # join into a single mapping (ids are unique per split in these examples)
                    ids_all = tr.ids + te.ids
                    X_raw_all = np.vstack([tr.X_raw, te.X_raw]) if tr.X_raw.size and te.X_raw.size else np.zeros((0,0))
                else:
                    all_ = loaded["all"]
                    ids_all = all_.ids
                    X_raw_all = all_.X_raw
                feature_names_raw = [f"v{i}" for i in range(X_raw_all.shape[1])]
                X_train_raw = _subset_by_ids(ids_all, X_raw_all, train_ids)
                X_test_raw = _subset_by_ids(ids_all, X_raw_all, test_ids)
            else:
                print(f"[EXPLAIN] dataset={ds_dir.name} method={method} unsupported input_type={input_type}; skipping")
                continue

            # Build wrappers that accept full RAW features and internally amplitude-encode to model input
            def predict_proba_full(X_raw_in: np.ndarray) -> np.ndarray:
                X_amp, _ = amplitude_encode_matrix(np.asarray(X_raw_in, dtype=float), pad_len=pad_len)
                return model.predict_proba(X_amp)

            def predict_prob1_full(X_raw_in: np.ndarray) -> np.ndarray:
                return predict_proba_full(X_raw_in)[:, 1]

            out_shap = mdir / "explain" / "shap"
            out_lime = mdir / "explain" / "lime"
            out_shap.mkdir(parents=True, exist_ok=True)
            out_lime.mkdir(parents=True, exist_ok=True)

            print(f"[EXPLAIN] dataset={ds_dir.name} method={method} input_type={input_type} pad_len={pad_len}")
            print(f"[EXPLAIN] train_rows={X_train_raw.shape[0]} test_rows={X_test_raw.shape[0]} raw_features={X_train_raw.shape[1]}")
            print(f"[EXPLAIN] writing -> {out_shap} and {out_lime}")

            # SHAP
            print(f"[SHAP] start method={method}")
            run_shap(
                predict_prob1_full=predict_prob1_full,
                X_train_raw=X_train_raw,
                X_test_raw=X_test_raw,
                feature_names_raw=feature_names_raw,
                out_dir=out_shap,
                max_features=int(args.max_features),
                max_test_samples=int(args.max_test_samples),
                background_samples=int(args.background_samples),
                max_evals=int(args.shap_max_evals),
            )
            print(f"[SHAP] done method={method}")

            # LIME
            print(f"[LIME] start method={method}")
            run_lime(
                predict_proba_full=predict_proba_full,
                X_train_raw=X_train_raw,
                X_test_raw=X_test_raw,
                feature_names_raw=feature_names_raw,
                class_names=[str(meta.get("labelmap", {}).get("classes", ["0","1"])[0]),
                             str(meta.get("labelmap", {}).get("classes", ["0","1"])[1])],
                out_dir=out_lime,
                max_features=int(args.max_features),
                max_test_samples=int(args.max_test_samples),
                num_features_explained=min(10, int(args.max_features)),
                num_samples=int(args.lime_num_samples),
            )
            print(f"[LIME] done method={method}")

        print(f"[EXPLAIN] dataset={ds_dir.name} done")

if __name__ == "__main__":
    main()
