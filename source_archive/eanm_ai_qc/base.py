from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
import numpy as np

class Model(Protocol):
    name: str
    def fit(self, X: np.ndarray, y01: np.ndarray) -> None: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, model_dir: Path, metadata: Dict[str, Any]) -> None: ...

@dataclass
class ModelArtifacts:
    model_dir: Path
    metadata: Dict[str, Any]
