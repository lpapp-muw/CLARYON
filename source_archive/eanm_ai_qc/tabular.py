from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from .encoding import amplitude_encode_matrix, AmplitudeEncodingInfo

@dataclass
class TabularDataset:
    name: str
    X_raw: np.ndarray
    X_amp: np.ndarray
    y_raw: Optional[np.ndarray]
    ids: List[str]
    feature_names: List[str]
    amp_info: AmplitudeEncodingInfo

def load_tabular_csv(
    path: Path,
    label_col: str = "label",
    id_col: Optional[str] = None,
    pad_len: Optional[int] = None,
    max_features: Optional[int] = None,
) -> TabularDataset:
    """Load a numeric tabular CSV.

    No statistical preprocessing is performed (no scaling/PCA).
    Minimal sanitation: NaN/Inf -> 0 to make amplitude encoding valid.
    """
    path = Path(path)
    df = pd.read_csv(path)

    if id_col is not None and id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
        df = df.drop(columns=[id_col])
    else:
        ids = [str(i) for i in range(len(df))]

    y_raw = None
    if label_col in df.columns:
        y_raw = df[label_col].to_numpy()
        Xdf = df.drop(columns=[label_col])
    else:
        Xdf = df

    # keep only numeric columns
    Xnum = Xdf.apply(pd.to_numeric, errors="coerce")
    # Optional feature cap (simulator convenience). Default: disabled.
    if max_features is not None and Xnum.shape[1] > max_features:
        Xnum = Xnum.iloc[:, :max_features]

    feature_names = list(Xnum.columns)
    X = Xnum.to_numpy(dtype=np.float64, copy=True)

    X_amp, info = amplitude_encode_matrix(X, pad_len=pad_len)

    return TabularDataset(
        name=path.stem,
        X_raw=X,
        X_amp=X_amp,
        y_raw=y_raw,
        ids=ids,
        feature_names=feature_names,
        amp_info=info,
    )
