#!/usr/bin/env python3
"""
preprocess_benchmark.py — E5: Feature Preprocessing for DEBI-NN Benchmarking

Converts raw benchmark datasets (features.csv + labels.csv) into DEBI-NN's
expected FDB/LDB format with proper preprocessing.

Preprocessing protocol (matches TabM ICLR 2025):
  - Numerical features:  Quantile normalization to standard normal
  - Categorical features: One-hot encoding
  - Binary features:     Map to {0, 1}
  - Missing values:      Median imputation (numerical), mode imputation (categorical)
  - Missing indicator:   Binary column added for features with >0 missing values
  - Labels:              Integer-encoded 0..K-1

Output format (DEBI-NN TabularDataFileIo compatible):
  - FDB.csv: Key;F0;F1;...;FN  (semicolon-separated, all numeric)
  - LDB.csv: Key;Label          (semicolon-separated, integer labels)
  - preprocessing_info.json:    Transform metadata for reproducibility
  - label_encoding.json:        Label name -> integer mapping

Usage:
    python preprocess_benchmark.py --input-dir benchmark_datasets --output-dir benchmark_preprocessed
    python preprocess_benchmark.py --input-dir benchmark_datasets --output-dir benchmark_preprocessed --dataset iris
    python preprocess_benchmark.py --input-dir benchmark_datasets --output-dir benchmark_preprocessed --no-quantile

Requirements:
    pip install pandas numpy scikit-learn joblib
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
import joblib

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Per-dataset configuration: target column + known categoricals
# ──────────────────────────────────────────────────────────────────────

DATASET_CONFIG = {
    # Tier 1: OpenML Standard
    "australian":           {"label_source": "labels", "categoricals": []},
    "blood-transfusion":    {"label_source": "labels", "categoricals": []},
    "credit-g":             {"label_source": "labels", "categoricals": [
        "checking_status", "credit_history", "purpose", "savings_status",
        "employment", "personal_status", "other_parties", "property_magnitude",
        "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"
    ]},
    "diabetes":             {"label_source": "labels", "categoricals": []},
    "kc1":                  {"label_source": "labels", "categoricals": []},
    "phoneme":              {"label_source": "labels", "categoricals": []},

    # Tier 2: Additional Standard
    "iris":                 {"label_source": "labels", "categoricals": []},
    "vehicle":              {"label_source": "labels", "categoricals": []},
    "segment":              {"label_source": "labels", "categoricals": []},
    "waveform-5000":        {"label_source": "labels", "categoricals": []},
    "steel-plates-fault":   {"label_source": "labels", "categoricals": []},
    "electricity":          {"label_source": "labels", "categoricals": []},
    "bank-marketing":       {"label_source": "labels", "categoricals": [
        "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "poutcome"
    ]},
    "adult":                {"label_source": "labels", "categoricals": [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]},

    # Tier 3: Medical
    "wisconsin-breast-cancer": {"label_source": "labels", "categoricals": []},
    "heart-failure":        {"label_source": "labels", "categoricals": []},
    "cervical-cancer":      {"label_source": "labels", "categoricals": []},
    "chronic-kidney-disease": {"label_source": "labels", "categoricals": [
        "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"
    ]},
    "spect-heart":          {"label_source": "labels", "categoricals": []},
    "hcc-survival":         {"label_source": "labels", "categoricals": []},
    "mammographic-mass":    {"label_source": "labels", "categoricals": []},
    "stroke-prediction":    {"label_source": "labels", "categoricals": [
        "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
    ]},

    # Tier 4: General domain
    "wine-quality":         {"label_source": "labels", "categoricals": []},
    "dry-bean":             {"label_source": "labels", "categoricals": []},
    "drug-classification":  {"label_source": "labels", "categoricals": [
        "Sex", "BP", "Cholesterol"
    ]},
    "fetal-health":         {"label_source": "labels", "categoricals": []},
    "rice-cammeo-osmancik": {"label_source": "labels", "categoricals": []},
    "mushroom":             {"label_source": "labels", "categoricals": "all"},
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def detect_categorical_columns(df, config_categoricals):
    """Detect categorical columns using config + dtype heuristics."""
    if config_categoricals == "all":
        return list(df.columns)

    cat_cols = list(config_categoricals) if config_categoricals else []

    # Also detect columns with object/string dtype not already listed
    for col in df.columns:
        if col in cat_cols:
            continue
        if df[col].dtype == object or df[col].dtype.name == "category":
            cat_cols.append(col)
        elif df[col].dtype == "bool":
            cat_cols.append(col)

    return cat_cols


def detect_binary_columns(df, exclude_cols):
    """Detect columns with exactly 2 unique non-null values (treat as binary)."""
    binary_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        nunique = df[col].dropna().nunique()
        if nunique == 2:
            binary_cols.append(col)
    return binary_cols


def preprocess_dataset(dataset_name, input_dir, output_dir, use_quantile=True):
    """Preprocess a single dataset into DEBI-NN format."""

    ds_in = os.path.join(input_dir, dataset_name)
    ds_out = ensure_dir(os.path.join(output_dir, dataset_name))

    # ── Load raw data ──
    features_path = os.path.join(ds_in, "features.csv")
    labels_path = os.path.join(ds_in, "labels.csv")
    raw_path = os.path.join(ds_in, "raw.csv")

    if os.path.exists(features_path) and os.path.exists(labels_path):
        X = pd.read_csv(features_path)
        y = pd.read_csv(labels_path)
        if y.shape[1] > 1:
            # Multiple label columns — use the first one
            target_col = y.columns[0]
            y = y[[target_col]]
        else:
            target_col = y.columns[0]
    elif os.path.exists(raw_path):
        raise ValueError(f"Dataset {dataset_name}: Only raw.csv found — needs manual target column specification.")
    else:
        raise FileNotFoundError(f"Dataset {dataset_name}: No features.csv/labels.csv found in {ds_in}")

    assert len(X) == len(y), f"Feature/label row count mismatch: {len(X)} vs {len(y)}"

    config = DATASET_CONFIG.get(dataset_name, {"label_source": "labels", "categoricals": []})

    # ── Step 1: Identify column types ──
    cat_cols = detect_categorical_columns(X, config.get("categoricals", []))
    # Filter to columns actually present in X
    cat_cols = [c for c in cat_cols if c in X.columns]

    num_cols = [c for c in X.columns if c not in cat_cols]

    # ── Step 2: Handle missing values ──
    missing_info = {}

    # Numerical: median imputation + missing indicator
    for col in num_cols:
        # Convert to numeric, coercing errors (handles "?" etc.)
        X[col] = pd.to_numeric(X[col], errors="coerce")
        n_missing = X[col].isna().sum()
        if n_missing > 0:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            # Add missing indicator column
            indicator_name = f"{col}_missing"
            X[indicator_name] = 0
            X.loc[X[col].isna(), indicator_name] = 1
            # Re-check after fillna (the indicator was set before fillna took effect)
            # Actually we need to detect before filling:
            missing_info[col] = {"n_missing": int(n_missing), "imputed_with": float(median_val)}

    # Rebuild: detect missing before imputing (fix the logic above)
    # Redo properly:
    X_raw = pd.read_csv(features_path) if os.path.exists(features_path) else None
    if X_raw is not None:
        indicator_cols_added = []
        for col in num_cols:
            X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
            n_missing = int(X_raw[col].isna().sum())
            if n_missing > 0:
                indicator_name = f"{col}_missing"
                X[indicator_name] = X_raw[col].isna().astype(int).values
                indicator_cols_added.append(indicator_name)
                median_val = X[col].median()  # X[col] was already filled above, so re-compute from raw
                median_val_raw = X_raw[col].median()
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(median_val_raw)
                missing_info[col] = {"n_missing": n_missing, "imputed_with": float(median_val_raw)}
        # These indicator columns are numerical (binary 0/1)
        num_cols = num_cols + indicator_cols_added

    # Categorical: mode imputation
    for col in cat_cols:
        X[col] = X[col].astype(str)  # Ensure string type
        X[col] = X[col].replace({"nan": np.nan, "None": np.nan, "?": np.nan, "": np.nan})
        n_missing = int(X[col].isna().sum())
        if n_missing > 0:
            mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else "unknown"
            X[col] = X[col].fillna(mode_val)
            missing_info[col] = {"n_missing": n_missing, "imputed_with": str(mode_val), "type": "categorical"}

    # ── Step 3: Encode categoricals ──
    encoding_info = {}

    # One-hot encode categorical columns
    if cat_cols:
        X_cat = pd.get_dummies(X[cat_cols], columns=cat_cols, prefix_sep="=", dtype=int)
        encoding_info["one_hot_columns"] = {col: [c for c in X_cat.columns if c.startswith(f"{col}=")] for col in cat_cols}
        # Drop original categorical columns, add one-hot
        X = X.drop(columns=cat_cols)
        X = pd.concat([X, X_cat], axis=1)
        # Update num_cols to include the new one-hot columns
        num_cols_after_ohe = [c for c in X.columns]
    else:
        num_cols_after_ohe = list(X.columns)

    # ── Step 4: Ensure all columns are numeric ──
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Final missing check — fill any remaining NaN with 0
    remaining_nan = X.isna().sum().sum()
    if remaining_nan > 0:
        print(f"    WARNING: {remaining_nan} remaining NaN values after imputation — filling with 0.0")
        X = X.fillna(0.0)

    # ── Step 5: Quantile normalization ──
    transform_obj = None
    if use_quantile and len(X) >= 30:  # Need minimum samples for quantile transform
        # Identify columns to transform (skip binary 0/1 columns like one-hot and missing indicators)
        cols_to_transform = []
        cols_to_skip = []
        for col in X.columns:
            unique_vals = X[col].unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 0.0, 1, 1.0}):
                cols_to_skip.append(col)
            else:
                cols_to_transform.append(col)

        if cols_to_transform:
            n_quantiles = min(1000, len(X))  # sklearn default but capped at n_samples
            qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                random_state=42,
                subsample=min(100000, len(X))
            )
            X[cols_to_transform] = qt.fit_transform(X[cols_to_transform])
            transform_obj = {
                "type": "QuantileTransformer",
                "columns": cols_to_transform,
                "skipped_columns": cols_to_skip,
                "n_quantiles": n_quantiles,
            }
            # Save the sklearn object for exact reproducibility
            joblib.dump(qt, os.path.join(ds_out, "quantile_transform.pkl"))
    elif use_quantile and len(X) < 30:
        print(f"    NOTE: Dataset too small for quantile transform ({len(X)} samples). Using raw values.")
        transform_obj = {"type": "none", "reason": "too_few_samples"}

    # ── Step 6: Encode labels ──
    y_series = y.iloc[:, 0]
    # Clean label values
    y_series = y_series.astype(str).str.strip()
    # Remove any NaN labels
    valid_mask = ~y_series.isin(["nan", "None", "", "NaN"])
    if not valid_mask.all():
        n_dropped = int((~valid_mask).sum())
        print(f"    WARNING: Dropping {n_dropped} samples with missing labels")
        X = X[valid_mask].reset_index(drop=True)
        y_series = y_series[valid_mask].reset_index(drop=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_series)
    label_mapping = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}

    # ── Step 7: Export as DEBI-NN format ──
    n_samples = len(X)
    n_features = X.shape[1]
    n_classes = len(le.classes_)

    # Create Key column: S0000, S0001, ...
    keys = [f"S{i:04d}" for i in range(n_samples)]

    # FDB.csv: Key;F0;F1;...;FN
    feature_names = [f"F{i}" for i in range(n_features)]
    fdb = pd.DataFrame(X.values, columns=feature_names)
    fdb.insert(0, "Key", keys)

    fdb_path = os.path.join(ds_out, "FDB.csv")
    fdb.to_csv(fdb_path, sep=";", index=False, float_format="%.8f")

    # LDB.csv: Key;Label
    ldb = pd.DataFrame({"Key": keys, "Label": y_encoded})
    ldb_path = os.path.join(ds_out, "LDB.csv")
    ldb.to_csv(ldb_path, sep=";", index=False)

    # ── Step 8: Save metadata ──
    original_feature_names = list(X.columns)

    preprocessing_info = {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "n_features_original": len(num_cols) + len(cat_cols),
        "n_features_after_preprocessing": n_features,
        "n_classes": n_classes,
        "n_categorical_original": len(cat_cols),
        "n_one_hot_columns": n_features - len(num_cols) if cat_cols else 0,
        "categorical_columns": cat_cols,
        "missing_value_handling": missing_info,
        "encoding": encoding_info,
        "normalization": transform_obj if transform_obj else {"type": "QuantileTransformer"},
        "label_column": target_col,
        "label_mapping": label_mapping,
        "feature_name_mapping": {f"F{i}": original_feature_names[i] for i in range(len(original_feature_names))},
        "output_format": {
            "fdb": "FDB.csv (semicolon-separated, Key;F0;F1;...;FN, all numeric float)",
            "ldb": "LDB.csv (semicolon-separated, Key;Label, integer labels 0..K-1)",
            "separator": ";",
            "key_format": "S0000..S{n-1}",
        },
    }

    with open(os.path.join(ds_out, "preprocessing_info.json"), "w") as f:
        json.dump(preprocessing_info, f, indent=2, default=str)

    with open(os.path.join(ds_out, "label_encoding.json"), "w") as f:
        json.dump(label_mapping, f, indent=2)

    return preprocessing_info


def main():
    parser = argparse.ArgumentParser(description="E5: Preprocess benchmark datasets for DEBI-NN")
    parser.add_argument("--input-dir", default="benchmark_datasets",
                        help="Directory containing raw downloaded datasets")
    parser.add_argument("--output-dir", default="benchmark_preprocessed",
                        help="Directory for preprocessed DEBI-NN-ready datasets")
    parser.add_argument("--dataset", default=None,
                        help="Process only this dataset (by folder name)")
    parser.add_argument("--no-quantile", action="store_true",
                        help="Skip quantile normalization (use raw numeric values)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = ensure_dir(args.output_dir)
    use_quantile = not args.no_quantile

    print(f"DEBI-NN Benchmark Preprocessor (E5)")
    print(f"Input:  {os.path.abspath(input_dir)}")
    print(f"Output: {os.path.abspath(output_dir)}")
    print(f"Quantile normalization: {'ON' if use_quantile else 'OFF'}")
    print(f"{'=' * 60}")

    # Determine which datasets to process
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted([
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d)) and d != "__pycache__"
        ])

    results = []
    errors = []

    for ds_name in datasets:
        print(f"\n  [{ds_name}] ", end="", flush=True)
        try:
            info = preprocess_dataset(ds_name, input_dir, output_dir, use_quantile)
            print(f"OK  ({info['n_samples']} samples, {info['n_features_after_preprocessing']} features, {info['n_classes']} classes)")
            results.append(info)
        except Exception as e:
            print(f"FAILED  ({e})")
            errors.append({"dataset": ds_name, "error": str(e)})

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Processed: {len(results)} / {len(datasets)} datasets")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors:
            print(f"  - {err['dataset']}: {err['error']}")

    # Summary table
    print(f"\n{'Dataset':<30} {'Samples':>8} {'Feat(raw)':>10} {'Feat(proc)':>11} {'Classes':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*10} {'-'*11} {'-'*8}")
    for r in results:
        print(f"{r['dataset']:<30} {r['n_samples']:>8} {r['n_features_original']:>10} {r['n_features_after_preprocessing']:>11} {r['n_classes']:>8}")

    # Save manifest
    manifest = {
        "protocol": "E5 v1.0",
        "preprocessing": "QuantileTransformer(output_distribution='normal')" if use_quantile else "none",
        "categorical_encoding": "one-hot",
        "missing_values": "median (numerical) + mode (categorical) + binary indicator",
        "output_format": "DEBI-NN FDB/LDB (semicolon-separated CSV)",
        "datasets": results,
        "errors": errors,
    }
    manifest_path = os.path.join(output_dir, "PREPROCESSING_MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
