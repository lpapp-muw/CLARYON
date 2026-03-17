"""PyRadiomics wrapper — feature extraction from NIfTI volumes.

New module. Wraps pyradiomics for radiomics feature extraction and merger
with existing tabular features.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_radiomics_features(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    case_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract radiomics features from a single image/mask pair.

    Args:
        image_path: Path to image volume (NIfTI).
        mask_path: Path to binary mask volume (NIfTI).
        config_path: Path to pyradiomics YAML config. If None, uses defaults.
        case_id: Optional case identifier for logging.

    Returns:
        Dict mapping feature names to values.
    """
    try:
        import radiomics
        from radiomics import featureextractor
    except ImportError as exc:
        raise ImportError(
            "pyradiomics is required for radiomics extraction. "
            "Install with: pip install pyradiomics"
        ) from exc

    image_path = str(Path(image_path))
    mask_path = str(Path(mask_path))

    params = {}
    if config_path is not None:
        params = {"params": str(Path(config_path))}

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    result = extractor.execute(image_path, mask_path)

    # Filter out diagnostics (keys starting with "diagnostics_")
    features = {
        k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
        for k, v in result.items()
        if not k.startswith("diagnostics_")
    }

    logger.debug(
        "Extracted %d features for %s", len(features),
        case_id or Path(image_path).stem,
    )
    return features


def extract_radiomics_batch(
    image_mask_pairs: List[Tuple[Union[str, Path], Union[str, Path]]],
    config_path: Optional[Union[str, Path]] = None,
    case_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract radiomics features from a batch of image/mask pairs.

    Args:
        image_mask_pairs: List of (image_path, mask_path) tuples.
        config_path: Path to pyradiomics YAML config.
        case_ids: Optional list of case identifiers (same length as pairs).

    Returns:
        DataFrame with case_id as index and feature names as columns.
    """
    if case_ids is None:
        case_ids = [Path(img).stem for img, _ in image_mask_pairs]

    rows = []
    for i, (img, mask) in enumerate(image_mask_pairs):
        cid = case_ids[i]
        try:
            features = extract_radiomics_features(img, mask, config_path, cid)
            features["case_id"] = cid
            rows.append(features)
        except Exception as e:
            logger.error("Failed to extract features for %s: %s", cid, e)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.set_index("case_id")

    # Convert to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Extracted radiomics features: %d samples × %d features", len(df), len(df.columns))
    return df


def merge_radiomics_with_tabular(
    tabular_X: np.ndarray,
    tabular_feature_names: List[str],
    radiomics_df: pd.DataFrame,
    tabular_keys: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Merge radiomics features with existing tabular features.

    Args:
        tabular_X: Existing feature matrix, shape (n_samples, n_tabular_features).
        tabular_feature_names: Column names for tabular features.
        radiomics_df: DataFrame from extract_radiomics_batch (case_id indexed).
        tabular_keys: Sample IDs matching tabular_X rows.

    Returns:
        Tuple of (merged_X, merged_feature_names).
    """
    tab_df = pd.DataFrame(tabular_X, columns=tabular_feature_names, index=tabular_keys)

    # Align radiomics to tabular order
    radiomics_aligned = radiomics_df.reindex(tabular_keys)

    # Fill missing radiomics with 0
    radiomics_aligned = radiomics_aligned.fillna(0.0)

    # Prefix radiomics columns to avoid name collisions
    radiomics_aligned.columns = [f"radiomics_{c}" for c in radiomics_aligned.columns]

    merged = pd.concat([tab_df, radiomics_aligned], axis=1)
    merged_X = merged.to_numpy(dtype=np.float64)
    merged_X = np.nan_to_num(merged_X, nan=0.0, posinf=0.0, neginf=0.0)

    return merged_X, list(merged.columns)
