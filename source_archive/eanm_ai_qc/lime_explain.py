from __future__ import annotations
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
import pandas as pd
from .utils import select_feature_indices_by_variance, subset_features

def run_lime(
    predict_proba_full: Callable[[np.ndarray], np.ndarray],
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    feature_names_raw: List[str],
    class_names: List[str],
    out_dir: Path,
    max_features: Optional[int] = 32,
    max_test_samples: int = 5,
    num_features_explained: int = 10,
    num_samples: int = 1000,
) -> None:
    """Model-agnostic LIME explanations.

    LIME runs in a reduced raw feature space; the model is evaluated by expanding
    to the full raw space with a training-mean baseline.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as e:
        (out_dir / "lime_SKIPPED.txt").write_text(f"LIME unavailable: {e}\n")
        return

    print(f"[LIME] starting: train={X_train_raw.shape} test={X_test_raw.shape} max_features={max_features} max_test_samples={max_test_samples} num_samples={num_samples}", flush=True)

    X_train_raw = np.asarray(X_train_raw, dtype=float)
    X_test_raw = np.asarray(X_test_raw, dtype=float)

    idx_full = select_feature_indices_by_variance(X_train_raw, max_features=max_features)
    Xtr_red, names_red = subset_features(X_train_raw, feature_names_raw, idx_full)
    Xte_red, _ = subset_features(X_test_raw, feature_names_raw, idx_full)

    baseline_full = np.mean(X_train_raw, axis=0)

    def predict_proba_red(x_red: np.ndarray) -> np.ndarray:
        x_red = np.asarray(x_red, dtype=float)
        if x_red.ndim == 1:
            x_red = x_red[None, :]
        Xfull = np.tile(baseline_full, (x_red.shape[0], 1))
        Xfull[:, idx_full] = x_red
        return predict_proba_full(Xfull)

    explainer = LimeTabularExplainer(
        training_data=Xtr_red,
        feature_names=names_red,
        class_names=class_names,
        mode="classification",
        discretize_continuous=False,
    )

    n = min(max_test_samples, Xte_red.shape[0])
    rows = []
    for i in range(n):
        print(f"[LIME] explaining sample {i+1}/{n}", flush=True)
        exp = explainer.explain_instance(
            data_row=Xte_red[i],
            predict_fn=predict_proba_red,
            num_features=min(num_features_explained, Xtr_red.shape[1]),
            num_samples=int(num_samples),
        )
        for feat, w in exp.as_list():
            rows.append({"sample_index": i, "feature": feat, "weight": float(w)})
    pd.DataFrame(rows).to_csv(out_dir / "lime_explanations.csv", index=False)

    pd.DataFrame({
        "reduced_feature_index": list(range(len(idx_full))),
        "original_feature_index": idx_full.tolist(),
        "feature_name": names_red,
    }).to_csv(out_dir / "lime_feature_index_map.csv", index=False)

    (out_dir / "lime_runinfo.json").write_text(
        pd.Series({
            "max_features": None if max_features is None else int(max_features),
            "max_test_samples": int(max_test_samples),
            "num_features_explained": int(num_features_explained),
            "num_samples": int(num_samples),
        }).to_json()
    )
