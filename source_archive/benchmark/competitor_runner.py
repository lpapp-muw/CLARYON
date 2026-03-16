#!/usr/bin/env python3
"""
competitor_runner.py — Run competitor methods on benchmark folds.

Reads the same TrainF/TrainL/TestF/TestL CSVs that DEBI-NN uses.
Outputs Predictions.csv in the same format for uniform aggregation:
    Key;Actual;Predicted;P0;P1;...;PC-1

Supported competitors (default configs for baseline run):
    CatBoost, XGBoost, LightGBM, TabPFN, MLP, TabM, RealMLP, ModernNCA
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from config import (
    RUNS_DIR, RESULTS_DIR, CSV_SEP, FLOAT_FMT,
    CV_SEEDS, N_FOLDS, DATASETS, LARGE_DATASET_THRESHOLD,
    DATASET_NAMES, COMPETITORS,
)

warnings.filterwarnings("ignore")


def load_fold_data(fold_dir):
    """Load TrainF/TrainL/TestF/TestL from a fold directory.

    Returns:
        X_train, y_train, X_test, y_test, test_keys, n_classes
    """
    train_f = pd.read_csv(os.path.join(fold_dir, "TrainF.csv"), sep=CSV_SEP)
    train_l = pd.read_csv(os.path.join(fold_dir, "TrainL.csv"), sep=CSV_SEP)
    test_f = pd.read_csv(os.path.join(fold_dir, "TestF.csv"), sep=CSV_SEP)
    test_l = pd.read_csv(os.path.join(fold_dir, "TestL.csv"), sep=CSV_SEP)

    test_keys = test_f["Key"].values
    feature_cols = [c for c in train_f.columns if c != "Key"]

    X_train = train_f[feature_cols].values.astype(np.float64)
    y_train = train_l["Label"].values.astype(int)
    X_test = test_f[feature_cols].values.astype(np.float64)
    y_test = test_l["Label"].values.astype(int)

    n_classes = max(y_train.max(), y_test.max()) + 1
    return X_train, y_train, X_test, y_test, test_keys, n_classes


def save_predictions(test_keys, y_test, y_pred, probs, out_path):
    """Save predictions in DEBI-NN Predictions.csv format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_classes = probs.shape[1]

    rows = []
    for i in range(len(test_keys)):
        row = {
            "Key": test_keys[i],
            "Actual": y_test[i],
            "Predicted": y_pred[i],
        }
        for c in range(n_classes):
            row[f"P{c}"] = f"{probs[i, c]:.8f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep=CSV_SEP, index=False)


# ── Competitor implementations ─────────────────────────────────────────

def run_catboost(X_train, y_train, X_test, n_classes, seed):
    """CatBoost with default config."""
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=1000,
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=0,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def run_xgboost(X_train, y_train, X_test, n_classes, seed):
    """XGBoost with default config."""
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=1000,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=seed,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def run_lightgbm(X_train, y_train, X_test, n_classes, seed):
    """LightGBM with default config."""
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def run_tabpfn(X_train, y_train, X_test, n_classes, seed):
    """TabPFN v2 (zero-shot foundation model)."""
    from tabpfn import TabPFNClassifier
    model = TabPFNClassifier()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def run_mlp(X_train, y_train, X_test, n_classes, seed):
    """scikit-learn MLP baseline."""
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=1000,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def run_tabm(X_train, y_train, X_test, n_classes, seed):
    """TabM placeholder — requires custom integration from yandex-research/tabm repo.

    TODO: Integrate TabM's official training script. For now, raises NotImplementedError
    so the harness skips it gracefully.
    """
    raise NotImplementedError(
        "TabM requires custom integration. Clone https://github.com/yandex-research/tabm "
        "and implement the adapter.")


def run_realmlp(X_train, y_train, X_test, n_classes, seed):
    """RealMLP via pytabkit."""
    try:
        from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
        model = RealMLP_TD_Classifier(random_state=seed, verbosity=0)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        preds = np.argmax(probs, axis=1)
        return preds, probs
    except ImportError:
        raise NotImplementedError(
            "RealMLP requires pytabkit: pip install pytabkit")


def run_modernnca(X_train, y_train, X_test, n_classes, seed):
    """ModernNCA placeholder — requires TALENT repo integration.

    TODO: Integrate from https://github.com/LAMDA-Tabular/TALENT.
    """
    raise NotImplementedError(
        "ModernNCA requires TALENT repo integration. "
        "Clone https://github.com/LAMDA-Tabular/TALENT and implement the adapter.")


# ── Dispatcher ─────────────────────────────────────────────────────────

COMPETITOR_FUNCS = {
    "CatBoost":   run_catboost,
    "XGBoost":    run_xgboost,
    "LightGBM":   run_lightgbm,
    "TabPFN":     run_tabpfn,
    "MLP":        run_mlp,
    "TabM":       run_tabm,
    "RealMLP":    run_realmlp,
    "ModernNCA":  run_modernnca,
}


def run_competitor_on_fold(method_name, fold_dir, out_dir, seed):
    """Run a single competitor on a single fold.

    Returns:
        dict with BACC, ACC, LogLoss, or error string.
    """
    func = COMPETITOR_FUNCS.get(method_name)
    if func is None:
        return {"error": f"Unknown method: {method_name}"}

    try:
        X_train, y_train, X_test, y_test, test_keys, n_classes = load_fold_data(fold_dir)
        preds, probs = func(X_train, y_train, X_test, n_classes, seed)

        # Ensure probs has correct shape (some methods may omit classes).
        if probs.shape[1] < n_classes:
            full_probs = np.zeros((len(preds), n_classes))
            full_probs[:, :probs.shape[1]] = probs
            probs = full_probs

        # Save predictions.
        pred_path = os.path.join(out_dir, "Predictions.csv")
        save_predictions(test_keys, y_test, preds, probs, pred_path)

        # Compute metrics.
        bacc = balanced_accuracy_score(y_test, preds)
        acc = np.mean(preds == y_test)

        # Clip probabilities for log_loss stability.
        probs_clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)
        probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
        ll = log_loss(y_test, probs_clipped, labels=list(range(n_classes)))

        return {"BACC": bacc, "ACC": acc, "LogLoss": ll}

    except NotImplementedError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def run_all_competitors(dataset_name, seed, fold_idx, methods=None):
    """Run all competitors on a single dataset/seed/fold.

    Returns:
        dict: method_name → result dict
    """
    if methods is None:
        methods = COMPETITORS

    fold_dir = os.path.join(RUNS_DIR, f"seed_{seed}", f"fold_{fold_idx}", dataset_name)
    results = {}

    for method in methods:
        out_dir = os.path.join(
            RESULTS_DIR, "competitors", method,
            f"seed_{seed}", f"fold_{fold_idx}", dataset_name)

        result = run_competitor_on_fold(method, fold_dir, out_dir, seed)
        results[method] = result

    return results


def get_fold_count(dataset_name):
    """Return number of folds for a dataset."""
    if DATASETS[dataset_name]["n"] > LARGE_DATASET_THRESHOLD:
        return 1
    return N_FOLDS


def main():
    parser = argparse.ArgumentParser(
        description="Run competitor baselines on benchmark folds")
    parser.add_argument("--dataset", default=None, help="Single dataset")
    parser.add_argument("--all", action="store_true", help="All 28 datasets")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Methods to run (default: all). Choices: {COMPETITORS}")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help=f"Seeds (default: {CV_SEEDS})")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        print("ERROR: Specify --dataset <name> or --all")
        return

    datasets = [args.dataset] if args.dataset else DATASET_NAMES
    seeds = args.seeds if args.seeds else CV_SEEDS
    methods = args.methods if args.methods else COMPETITORS

    print(f"Competitor Runner | methods={methods}")
    print(f"  Seeds: {seeds}")
    print(f"{'=' * 60}")

    for ds in datasets:
        if ds not in DATASETS:
            print(f"  [{ds}] SKIPPED")
            continue

        n_folds = get_fold_count(ds)
        for seed in seeds:
            for fold_idx in range(n_folds):
                print(f"  [{ds}] seed={seed} fold={fold_idx} ", end="", flush=True)
                results = run_all_competitors(ds, seed, fold_idx, methods)

                status_parts = []
                for method, res in results.items():
                    if "error" in res:
                        status_parts.append(f"{method}:SKIP")
                    else:
                        status_parts.append(f"{method}:{res['BACC']:.3f}")
                print("  ".join(status_parts))

    print("\nDone.")


if __name__ == "__main__":
    main()
