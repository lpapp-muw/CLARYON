from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

def select_feature_indices_by_variance(X: np.ndarray, max_features: Optional[int]) -> np.ndarray:
    n_features = X.shape[1]
    if max_features is None or max_features >= n_features:
        return np.arange(n_features, dtype=int)
    v = np.var(X, axis=0)
    idx = np.argsort(v)[::-1][:max_features]
    idx = np.sort(idx)
    return idx

def subset_features(X: np.ndarray, feature_names: List[str], idx: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    Xs = X[:, idx]
    names = [feature_names[i] for i in idx.tolist()]
    return Xs, names
