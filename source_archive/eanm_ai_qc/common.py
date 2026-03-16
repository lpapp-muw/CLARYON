from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

def to_py_scalar(v: Any) -> Any:
    try:
        import numpy as _np
        if isinstance(v, _np.generic):
            return v.item()
    except Exception:
        pass
    return v

@dataclass
class BinaryLabelMapper:
    classes: List[Any]
    to_int: Dict[Any, int]
    to_label: Dict[int, Any]

    @staticmethod
    def fit(y: List[Any]) -> "BinaryLabelMapper":
        uniq = pd.unique(pd.Series(y))
        if len(uniq) != 2:
            raise ValueError(f"Binary classification required; got {len(uniq)} unique labels: {list(uniq)}")
        classes = sorted([to_py_scalar(v) for v in list(uniq)], key=lambda x: str(x))
        to_int = {classes[0]: 0, classes[1]: 1}
        to_label = {0: classes[0], 1: classes[1]}
        return BinaryLabelMapper(classes=classes, to_int=to_int, to_label=to_label)

    def transform(self, y: List[Any]) -> np.ndarray:
        return np.array([self.to_int[to_py_scalar(v)] for v in y], dtype=np.int64)

    def inverse_transform(self, y01: np.ndarray) -> List[Any]:
        return [self.to_label[int(v)] for v in y01.tolist()]

    def to_json(self) -> Dict[str, Any]:
        return {"classes": [to_py_scalar(self.classes[0]), to_py_scalar(self.classes[1])]}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "BinaryLabelMapper":
        classes = [d["classes"][0], d["classes"][1]]
        to_int = {classes[0]: 0, classes[1]: 1}
        to_label = {0: classes[0], 1: classes[1]}
        return BinaryLabelMapper(classes=classes, to_int=to_int, to_label=to_label)
