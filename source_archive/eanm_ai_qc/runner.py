from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .io.common import BinaryLabelMapper
from .io.tabular import load_tabular_csv
from .io.nifti import load_nifti_dataset, load_nifti_for_inference
from .io.encoding import amplitude_encode_matrix
from .metrics import binary_metrics, select_threshold_balanced_accuracy
from .models import PLAmplitudeKernelSVM, PLQCNN_MUW, PLQCNN_Alt
from .explain import run_shap, run_lime

SUPPORTED_METHODS = {
    "pl_kernel_svm": PLAmplitudeKernelSVM,
    "pl_qcnn_muw": PLQCNN_MUW,
    "pl_qcnn_alt": PLQCNN_Alt,
}

def dataset_base_name(path: Path) -> str:
    path = Path(path)
    return path.name if path.is_dir() else path.stem

def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(msg: str) -> None:
    print(msg, flush=True)

@dataclass
class RunConfig:
    input_path: Path
    input_type: str  # tabular | nifti | auto
    infer_path: Optional[Path]
    methods: List[str]
    results_dir: Path
    seed: int
    test_size: float
    shots: Optional[int]

    # qcnn
    qcnn_epochs: int
    qcnn_lr: float
    qcnn_batch_size: Optional[int]
    qcnn_init_scale: float
    qcnn_dense_layers: int
    qcnn_layers: Optional[int]

    # controls
    max_samples_per_method: Optional[int]
    max_features_qcnn_tabular: Optional[int]

    # explainability
    explain: bool
    explain_max_features: Optional[int]
    explain_max_test_samples: int
    explain_background_samples: int

def _write_predictions(path: Path, ids: List[str], prob1: np.ndarray, pred01: np.ndarray, true01: Optional[np.ndarray], labelmap: BinaryLabelMapper) -> None:
    df = pd.DataFrame({"id": ids, "prob_1": prob1, "pred_01": pred01})
    df["pred_label"] = labelmap.inverse_transform(pred01)
    if true01 is not None:
        df["true_label"] = labelmap.inverse_transform(true01)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def run_experiment(cfg: RunConfig) -> Path:
    results_dir = ensure_dir(cfg.results_dir)
    ds_name = dataset_base_name(cfg.input_path)
    ds_dir = ensure_dir(results_dir / ds_name)

    log(f"[RUN] dataset={ds_name} input_type={cfg.input_type} input={cfg.input_path}")
    log(f"[RUN] methods={cfg.methods} explain={cfg.explain}")

    # --- Load data ---
    if cfg.input_type == "auto":
        it = "nifti" if Path(cfg.input_path).is_dir() else "tabular"
    else:
        it = cfg.input_type

    log(f"[RUN] resolved_input_type={it}")
    infer = None

    if it == "tabular":
        ds = load_tabular_csv(cfg.input_path, label_col="label", pad_len=None, max_features=None)
        if ds.y_raw is None:
            raise ValueError("Tabular input must contain a 'label' column for training/evaluation.")
        labelmap = BinaryLabelMapper.fit(ds.y_raw.tolist())
        y01 = labelmap.transform(ds.y_raw.tolist())

        idx_all = np.arange(ds.X_raw.shape[0])
        idx_tr, idx_te = train_test_split(idx_all, test_size=cfg.test_size, random_state=cfg.seed, stratify=y01)

        Xraw_tr = ds.X_raw[idx_tr]
        Xraw_te = ds.X_raw[idx_te]
        Xamp_tr = ds.X_amp[idx_tr]
        Xamp_te = ds.X_amp[idx_te]
        ytr = y01[idx_tr]
        yte = y01[idx_te]
        id_tr = [ds.ids[i] for i in idx_tr.tolist()]
        id_te = [ds.ids[i] for i in idx_te.tolist()]
        feature_names_raw = ds.feature_names
        amp_info = ds.amp_info
        log(f"[RUN][TABULAR] samples={ds.X_raw.shape[0]} raw_features={ds.X_raw.shape[1]} pad_len={amp_info.pad_len} n_qubits={amp_info.n_qubits}")

        if cfg.infer_path is not None:
            ds_inf = load_tabular_csv(cfg.infer_path, label_col="label", pad_len=amp_info.pad_len, max_features=None)
            infer = {"X_raw": ds_inf.X_raw, "X_amp": ds_inf.X_amp, "ids": ds_inf.ids}

    elif it == "nifti":
        loaded = load_nifti_dataset(cfg.input_path)
        if "train" in loaded:
            tr = loaded["train"]
            te = loaded["test"]
            labelmap = BinaryLabelMapper.fit(tr.y_raw.tolist() + te.y_raw.tolist())
            ytr = labelmap.transform(tr.y_raw.tolist())
            yte = labelmap.transform(te.y_raw.tolist())
            Xraw_tr, Xamp_tr, id_tr = tr.X_raw, tr.X_amp, tr.ids
            Xraw_te, Xamp_te, id_te = te.X_raw, te.X_amp, te.ids
            amp_info = loaded["amp_info"]
        else:
            all_ = loaded["all"]
            labelmap = BinaryLabelMapper.fit(all_.y_raw.tolist())
            y01 = labelmap.transform(all_.y_raw.tolist())
            idx_all = np.arange(all_.X_raw.shape[0])
            idx_tr, idx_te = train_test_split(idx_all, test_size=cfg.test_size, random_state=cfg.seed, stratify=y01)
            Xraw_tr, Xraw_te = all_.X_raw[idx_tr], all_.X_raw[idx_te]
            Xamp_tr, Xamp_te = all_.X_amp[idx_tr], all_.X_amp[idx_te]
            ytr, yte = y01[idx_tr], y01[idx_te]
            id_tr = [all_.ids[i] for i in idx_tr.tolist()]
            id_te = [all_.ids[i] for i in idx_te.tolist()]
            amp_info = loaded["amp_info"]

        # raw feature names are voxel indices
        feature_names_raw = [f"v{i}" for i in range(Xraw_tr.shape[1])]
        log(f"[RUN][NIFTI] train={Xraw_tr.shape[0]} test={Xraw_te.shape[0]} raw_len={Xraw_tr.shape[1]} pad_len={amp_info.pad_len} n_qubits={amp_info.n_qubits} masks={'on' if loaded.get('preprocess', {}).get('mask_pattern', None) else 'off'}")

        if cfg.infer_path is not None:
            pp = loaded["preprocess"]
            Xraw_inf, Xamp_inf, ids_inf, _ = load_nifti_for_inference(
                cfg.infer_path,
                pet_pattern=pp.get("pet_pattern", "*PET*.nii*"),
                mask_pattern=pp.get("mask_pattern", None),
                pad_len=amp_info.pad_len,
            )
            infer = {"X_raw": Xraw_inf, "X_amp": Xamp_inf, "ids": ids_inf}

    else:
        raise ValueError("input_type must be tabular|nifti|auto")

    # --- Run methods ---
    rows: List[Dict[str, Any]] = []
    summary_csv = results_dir / f"{ds_name}__results.csv"

    for method in cfg.methods:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: {method}. Supported: {sorted(SUPPORTED_METHODS)}")

        mdir = ensure_dir(ds_dir / method)
        model_dir = ensure_dir(mdir / "model")
        pred_dir = ensure_dir(mdir / "predictions")
        explain_dir = ensure_dir(mdir / "explain")

        log(f"[RUN] method={method} start")

        # subsample training rows for runtime (applies to raw and amplitude consistently)
        Xraw_tr_m, Xamp_tr_m, ytr_m, id_tr_m = Xraw_tr, Xamp_tr, ytr, id_tr
        if cfg.max_samples_per_method is not None and Xamp_tr.shape[0] > cfg.max_samples_per_method:
            rng = np.random.default_rng(cfg.seed)
            sel = np.sort(rng.choice(Xamp_tr.shape[0], size=cfg.max_samples_per_method, replace=False))
            Xraw_tr_m = Xraw_tr[sel]
            Xamp_tr_m = Xamp_tr[sel]
            ytr_m = ytr[sel]
            id_tr_m = [id_tr[i] for i in sel.tolist()]

        # Optional feature cap for tabular->QCNN (controls qubits). This reloads the dataset in raw space.
        if it == "tabular" and method in {"pl_qcnn_muw", "pl_qcnn_alt"} and cfg.max_features_qcnn_tabular is not None:
            ds_cap = load_tabular_csv(cfg.input_path, label_col="label", pad_len=None, max_features=cfg.max_features_qcnn_tabular)
            labelmap = BinaryLabelMapper.fit(ds_cap.y_raw.tolist())
            y01_cap = labelmap.transform(ds_cap.y_raw.tolist())

            idx_all = np.arange(ds_cap.X_raw.shape[0])
            idx_tr, idx_te = train_test_split(idx_all, test_size=cfg.test_size, random_state=cfg.seed, stratify=y01_cap)

            Xraw_tr = ds_cap.X_raw[idx_tr]
            Xraw_te = ds_cap.X_raw[idx_te]
            Xamp_tr = ds_cap.X_amp[idx_tr]
            Xamp_te = ds_cap.X_amp[idx_te]
            ytr = y01_cap[idx_tr]
            yte = y01_cap[idx_te]
            id_tr = [ds_cap.ids[i] for i in idx_tr.tolist()]
            id_te = [ds_cap.ids[i] for i in idx_te.tolist()]
            feature_names_raw = ds_cap.feature_names
            amp_info = ds_cap.amp_info

            # subsample after cap
            Xraw_tr_m, Xamp_tr_m, ytr_m, id_tr_m = Xraw_tr, Xamp_tr, ytr, id_tr
            if cfg.max_samples_per_method is not None and Xamp_tr.shape[0] > cfg.max_samples_per_method:
                rng = np.random.default_rng(cfg.seed)
                sel = np.sort(rng.choice(Xamp_tr.shape[0], size=cfg.max_samples_per_method, replace=False))
                Xraw_tr_m = Xraw_tr[sel]
                Xamp_tr_m = Xamp_tr[sel]
                ytr_m = ytr[sel]
                id_tr_m = [id_tr[i] for i in sel.tolist()]

            if cfg.infer_path is not None:
                ds_inf_cap = load_tabular_csv(cfg.infer_path, label_col="label", pad_len=amp_info.pad_len, max_features=cfg.max_features_qcnn_tabular)
                infer = {"X_raw": ds_inf_cap.X_raw, "X_amp": ds_inf_cap.X_amp, "ids": ds_inf_cap.ids}

        # instantiate model
        log(f"[RUN] method={method} train_used={Xamp_tr_m.shape[0]} test={Xamp_te.shape[0]} pad_len={amp_info.pad_len} n_qubits={amp_info.n_qubits}")
        ModelCls = SUPPORTED_METHODS[method]
        if method == "pl_kernel_svm":
            model = ModelCls(n_qubits=amp_info.n_qubits, pad_len=amp_info.pad_len, seed=cfg.seed, shots=cfg.shots)
        elif method == "pl_qcnn_muw":
            model = ModelCls(n_qubits=amp_info.n_qubits, pad_len=amp_info.pad_len, seed=cfg.seed, shots=cfg.shots, n_layers=cfg.qcnn_layers, init_scale=cfg.qcnn_init_scale)
        else:
            model = ModelCls(n_qubits=amp_info.n_qubits, pad_len=amp_info.pad_len, seed=cfg.seed, shots=cfg.shots, n_layers=cfg.qcnn_layers, dense_layers=cfg.qcnn_dense_layers, init_scale=cfg.qcnn_init_scale)

        # train
        if method in {"pl_qcnn_muw", "pl_qcnn_alt"}:
            model.fit(Xamp_tr_m, ytr_m, epochs=cfg.qcnn_epochs, lr=cfg.qcnn_lr, batch_size=cfg.qcnn_batch_size)
        else:
            model.fit(Xamp_tr_m, ytr_m)

        log(f"[RUN] method={method} training_done")

        # predict train/test
        prob_tr = model.predict_proba(Xamp_tr_m)[:, 1]
        prob_te = model.predict_proba(Xamp_te)[:, 1]

        # IMPORTANT: many quantum models output poorly-calibrated probabilities around ~0.5.
        # We therefore select a decision threshold on the TRAIN split only, maximizing balanced accuracy.
        thr = select_threshold_balanced_accuracy(ytr_m, prob_tr, default=0.5)

        pred_tr = (prob_tr >= thr).astype(int)
        pred_te = (prob_te >= thr).astype(int)

        met_tr = binary_metrics(ytr_m, prob_tr, threshold=thr)
        met_te = binary_metrics(yte, prob_te, threshold=thr)

        log(
            f"[RUN] method={method} decision_threshold(train_opt)={thr:.3f} "
            f"test_accuracy={met_te.get('accuracy'):.4f} test_auc={met_te.get('auc'):.4f} "
            f"test_bal_acc={met_te.get('balanced_accuracy'):.4f}"
        )

        _write_predictions(pred_dir / "train.csv", id_tr_m, prob_tr, pred_tr, ytr_m, labelmap)
        _write_predictions(pred_dir / "test.csv", id_te, prob_te, pred_te, yte, labelmap)

        if infer is not None:
            prob_inf = model.predict_proba(infer["X_amp"])[:, 1]
            pred_inf = (prob_inf >= thr).astype(int)
            _write_predictions(pred_dir / "infer.csv", infer["ids"], prob_inf, pred_inf, None, labelmap)

        meta = {
            "dataset": ds_name,
            "input_path": str(cfg.input_path),
            "input_type": it,
            "method": method,
            "decision_threshold": float(thr),
            "seed": cfg.seed,
            "shots": cfg.shots,
            "decision_threshold": float(thr),
            "n_qubits": amp_info.n_qubits,
            "pad_len": amp_info.pad_len,
            "amp_n_features_in": amp_info.n_features_in,
            "labelmap": labelmap.to_json(),
            "notes": {
                "no_statistical_preprocessing": True,
                "encoding": "amplitude padding + L2 normalization (required); NaN/Inf replaced with 0",
            },
            "hyperparams": {
                "qcnn_epochs": cfg.qcnn_epochs if method in {"pl_qcnn_muw", "pl_qcnn_alt"} else None,
                "qcnn_lr": cfg.qcnn_lr if method in {"pl_qcnn_muw", "pl_qcnn_alt"} else None,
                "qcnn_batch_size": cfg.qcnn_batch_size if method in {"pl_qcnn_muw", "pl_qcnn_alt"} else None,
                "qcnn_init_scale": cfg.qcnn_init_scale if method in {"pl_qcnn_muw", "pl_qcnn_alt"} else None,
                "qcnn_dense_layers": cfg.qcnn_dense_layers if method == "pl_qcnn_alt" else None,
                "qcnn_layers": cfg.qcnn_layers if method in {"pl_qcnn_muw", "pl_qcnn_alt"} else None,
                "max_features_qcnn_tabular": cfg.max_features_qcnn_tabular if (it == "tabular" and method in {"pl_qcnn_muw", "pl_qcnn_alt"}) else None,
            },
        }
        model.save(model_dir, meta)

        # explainability on TEST set (raw feature space)
        if cfg.explain:
            log(f"[RUN] method={method} explainability=ON (SHAP+LIME); this can be slow")

            def predict_proba_full(X_raw_in: np.ndarray) -> np.ndarray:
                X_amp_in, _ = amplitude_encode_matrix(np.asarray(X_raw_in, dtype=float), pad_len=amp_info.pad_len)
                return model.predict_proba(X_amp_in)

            def predict_prob1_full(X_raw_in: np.ndarray) -> np.ndarray:
                return predict_proba_full(X_raw_in)[:, 1]

            run_shap(
                predict_prob1_full=predict_prob1_full,
                X_train_raw=Xraw_tr_m,
                X_test_raw=Xraw_te,
                feature_names_raw=feature_names_raw,
                out_dir=explain_dir / "shap",
                max_features=cfg.explain_max_features,
                max_test_samples=cfg.explain_max_test_samples,
                background_samples=cfg.explain_background_samples,
            )

            run_lime(
                predict_proba_full=predict_proba_full,
                X_train_raw=Xraw_tr_m,
                X_test_raw=Xraw_te,
                feature_names_raw=feature_names_raw,
                class_names=[str(labelmap.classes[0]), str(labelmap.classes[1])],
                out_dir=explain_dir / "lime",
                max_features=cfg.explain_max_features,
                max_test_samples=min(cfg.explain_max_test_samples, 5),
                num_features_explained=min(10, cfg.explain_max_features or 10),
            )
        else:
            log(f"[RUN] method={method} explainability=OFF (--no-explain)")
        row = {
            "dataset": ds_name,
            "method": method,
            "decision_threshold": float(thr),
            **{f"train_{k}": v for k, v in met_tr.items()},
            **{f"test_{k}": v for k, v in met_te.items()},
            "n_train_used": int(len(ytr_m)),
            "n_test": int(len(yte)),
            "n_qubits": int(amp_info.n_qubits),
            "pad_len": int(amp_info.pad_len),
        }
        rows.append(row)

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(summary_csv, index=False)
    df_sum.to_csv(ds_dir / f"{ds_name}__results.csv", index=False)

    log(f"[RUN] wrote_summary={summary_csv}")
    return summary_csv
