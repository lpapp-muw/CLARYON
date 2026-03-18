"""Tabular preprocessing — imputation, one-hot encoding, quantile normalization.

Ported from [B] preprocess_benchmark.py. Generalized beyond DEBI-NN format.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

logger = logging.getLogger(__name__)


def fit_zscore(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute z-score parameters from training data.

    Args:
        X_train: Training feature matrix, shape (n_samples, n_features).

    Returns:
        (mean, std) arrays each of shape (n_features,).
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    return mean, std


def apply_zscore(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply z-score normalization using pre-computed parameters.

    Args:
        X: Feature matrix, shape (n_samples, n_features).
        mean: Per-feature mean from training data.
        std: Per-feature std from training data.

    Returns:
        Z-score normalized array.
    """
    return (X - mean) / np.maximum(std, 1e-12)


@dataclass
class PreprocessingResult:
    """Result of tabular preprocessing.

    Attributes:
        X: Preprocessed feature matrix, shape (n_samples, n_features).
        feature_names: Column names after preprocessing.
        metadata: Preprocessing metadata for reproducibility.
    """

    X: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def detect_categorical_columns(
    df: pd.DataFrame,
    known_categoricals: Optional[List[str]] = None,
) -> List[str]:
    """Detect categorical columns using config hints + dtype heuristics.

    Args:
        df: Feature DataFrame.
        known_categoricals: Columns known to be categorical. Use ``"all"``
            as a special value via the caller to treat all columns as categorical.

    Returns:
        List of categorical column names.
    """
    cat_cols = list(known_categoricals) if known_categoricals else []

    for col in df.columns:
        if col in cat_cols:
            continue
        if df[col].dtype == object or df[col].dtype.name in ("category", "string", "str"):
            cat_cols.append(col)
        elif str(df[col].dtype) == "str":
            cat_cols.append(col)
        elif df[col].dtype == "bool":
            cat_cols.append(col)

    # Filter to actually present columns
    return [c for c in cat_cols if c in df.columns]


def preprocess_tabular(
    X: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    use_quantile: bool = True,
    quantile_seed: int = 42,
    missing_indicator: bool = True,
) -> PreprocessingResult:
    """Apply the full tabular preprocessing pipeline.

    Steps:
        1. Detect categorical vs numerical columns
        2. Median imputation (numerical) + mode imputation (categorical)
        3. Optionally add binary missing indicators
        4. One-hot encode categoricals
        5. Quantile normalization (skipping binary 0/1 columns)

    Args:
        X: Feature DataFrame (no label, no ID column).
        categorical_columns: Known categorical column names.
        use_quantile: Whether to apply QuantileTransformer.
        quantile_seed: Random seed for QuantileTransformer.
        missing_indicator: Whether to add binary missing indicator columns.

    Returns:
        PreprocessingResult with processed features and metadata.
    """
    X = X.copy()
    metadata: Dict[str, Any] = {}

    # Step 1: Identify column types
    cat_cols = detect_categorical_columns(X, categorical_columns)
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Step 2: Imputation
    imputation_info: Dict[str, Any] = {}
    indicator_cols: List[str] = []

    # Numerical: median imputation
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        n_missing = int(X[col].isna().sum())
        if n_missing > 0:
            median_val = X[col].median()
            if missing_indicator:
                ind_name = f"{col}_missing"
                X[ind_name] = X[col].isna().astype(int)
                indicator_cols.append(ind_name)
            X[col] = X[col].fillna(median_val)
            imputation_info[col] = {"n_missing": n_missing, "imputed_with": float(median_val)}

    # Categorical: mode imputation
    for col in cat_cols:
        X[col] = X[col].astype(str)
        X[col] = X[col].replace({"nan": np.nan, "None": np.nan, "?": np.nan, "": np.nan})
        n_missing = int(X[col].isna().sum())
        if n_missing > 0:
            mode_vals = X[col].mode()
            mode_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "unknown"
            X[col] = X[col].fillna(mode_val)
            imputation_info[col] = {
                "n_missing": n_missing, "imputed_with": str(mode_val), "type": "categorical"
            }

    # Step 3: One-hot encode categoricals
    encoding_info: Dict[str, List[str]] = {}
    if cat_cols:
        X_cat = pd.get_dummies(X[cat_cols], columns=cat_cols, prefix_sep="=", dtype=int)
        encoding_info = {
            col: [c for c in X_cat.columns if c.startswith(f"{col}=")]
            for col in cat_cols
        }
        X = X.drop(columns=cat_cols)
        X = pd.concat([X, X_cat], axis=1)

    # Update num_cols with indicators
    num_cols = num_cols + indicator_cols

    # Step 4: Ensure all numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    remaining_nan = int(X.isna().sum().sum())
    if remaining_nan > 0:
        logger.warning("%d remaining NaN values after imputation — filling with 0.0", remaining_nan)
        X = X.fillna(0.0)

    # Step 5: Quantile normalization
    transform_info: Optional[Dict[str, Any]] = None
    if use_quantile and len(X) >= 30:
        cols_to_transform = []
        cols_to_skip = []
        for col in X.columns:
            unique_vals = X[col].unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 0.0, 1, 1.0}):
                cols_to_skip.append(col)
            else:
                cols_to_transform.append(col)

        if cols_to_transform:
            n_quantiles = min(1000, len(X))
            qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                random_state=quantile_seed,
                subsample=min(100000, len(X)),
            )
            X[cols_to_transform] = qt.fit_transform(X[cols_to_transform])
            transform_info = {
                "type": "QuantileTransformer",
                "columns_transformed": cols_to_transform,
                "columns_skipped": cols_to_skip,
                "n_quantiles": n_quantiles,
            }

    feature_names = list(X.columns)
    X_arr = X.to_numpy(dtype=np.float64)

    metadata = {
        "n_features_original": len(num_cols) + len(cat_cols) - len(indicator_cols),
        "n_features_after": len(feature_names),
        "categorical_columns": cat_cols,
        "indicator_columns": indicator_cols,
        "imputation": imputation_info,
        "encoding": encoding_info,
        "normalization": transform_info,
    }

    return PreprocessingResult(X=X_arr, feature_names=feature_names, metadata=metadata)
