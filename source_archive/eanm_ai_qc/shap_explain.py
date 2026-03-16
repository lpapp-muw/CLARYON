from __future__ import annotations
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
import pandas as pd
from .utils import select_feature_indices_by_variance, subset_features

def run_shap(
    predict_prob1_full: Callable[[np.ndarray], np.ndarray],
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    feature_names_raw: List[str],
    out_dir: Path,
    max_features: Optional[int] = 32,
    max_test_samples: int = 10,
    background_samples: int = 20,
    max_evals: Optional[int] = None,
) -> None:
    """Model-agnostic SHAP with a permutation explainer.

    The explainer operates in a *reduced raw feature space* for feasibility, but the
    model is always called on the full raw feature vector via baseline expansion.

    Saved outputs (in out_dir):
    - shap_values.npy
    - shap_summary.csv (mean |SHAP| per feature)
    - shap_samples.csv (per-sample SHAP values for explained samples)
    - shap_feature_index_map.csv (reduced feature index -> original feature index/name)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap
    except Exception as e:
        (out_dir / "shap_SKIPPED.txt").write_text(f"SHAP unavailable: {e}\n")
        return

    print(f"[SHAP] starting permutation explainer: train={X_train_raw.shape} test={X_test_raw.shape} max_features={max_features} max_test_samples={max_test_samples} background_samples={background_samples}", flush=True)

    X_train_raw = np.asarray(X_train_raw, dtype=float)
    X_test_raw = np.asarray(X_test_raw, dtype=float)

    idx_full = select_feature_indices_by_variance(X_train_raw, max_features=max_features)
    Xtr_red, names_red = subset_features(X_train_raw, feature_names_raw, idx_full)
    Xte_red, _ = subset_features(X_test_raw, feature_names_raw, idx_full)

    baseline_full = np.mean(X_train_raw, axis=0)

    def f_red(x_red: np.ndarray) -> np.ndarray:
        x_red = np.asarray(x_red, dtype=float)
        if x_red.ndim == 1:
            x_red = x_red[None, :]
        Xfull = np.tile(baseline_full, (x_red.shape[0], 1))
        Xfull[:, idx_full] = x_red
        return predict_prob1_full(Xfull)

    n_bg = min(background_samples, Xtr_red.shape[0])
    n_te = min(max_test_samples, Xte_red.shape[0])
    bg = Xtr_red[:n_bg]
    ex = Xte_red[:n_te]

    explainer = shap.PermutationExplainer(f_red, bg)

    # Limit the number of model evaluations for runtime control.
    # If not provided, choose a small multiple of feature count.
    if max_evals is None:
        # a conservative default; higher values increase fidelity and runtime
        max_evals = int(min(2048, 10 * max(1, ex.shape[1])))

    try:
        shap_out = explainer(ex, max_evals=max_evals)
    except TypeError:
        # older SHAP API without max_evals
        shap_out = explainer(ex)

    shap_vals = shap_out.values  # (n_te, n_features_red)
    np.save(out_dir / "shap_values.npy", shap_vals)

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    pd.DataFrame({"feature": names_red, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)        .to_csv(out_dir / "shap_summary.csv", index=False)

    rows = []
    for i in range(shap_vals.shape[0]):
        for j, fn in enumerate(names_red):
            rows.append({"sample_index": i, "feature": fn, "shap_value": float(shap_vals[i, j])})
    pd.DataFrame(rows).to_csv(out_dir / "shap_samples.csv", index=False)

    pd.DataFrame({
        "reduced_feature_index": list(range(len(idx_full))),
        "original_feature_index": idx_full.tolist(),
        "feature_name": names_red,
    }).to_csv(out_dir / "shap_feature_index_map.csv", index=False)

    (out_dir / "shap_runinfo.json").write_text(
        pd.Series({
            "max_features": None if max_features is None else int(max_features),
            "max_test_samples": int(max_test_samples),
            "background_samples": int(background_samples),
            "max_evals": int(max_evals),
        }).to_json()
    )
