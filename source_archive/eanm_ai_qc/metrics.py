from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def select_threshold_balanced_accuracy(
    y_true: np.ndarray,
    prob1: np.ndarray,
    default: float = 0.5,
) -> float:
    """Select a decision threshold using *training* data only.

    Strategy:
    - Use ROC curve thresholds and choose the one maximizing Youden's J = TPR - FPR.
      This is equivalent to maximizing balanced accuracy.
    - Tie-breaker: choose the threshold closest to `default` (usually 0.5).
    - Returns a value in [0, 1]. Falls back to `default` if undefined.

    Deterministic given y_true/prob1.
    """
    y_true = np.asarray(y_true).astype(int)
    prob1 = np.asarray(prob1).astype(float)

    if y_true.size == 0 or prob1.size == 0:
        return float(default)
    if len(np.unique(y_true)) < 2:
        # no negative/positive separation possible
        return float(default)

    try:
        fpr, tpr, thr = roc_curve(y_true, prob1)
    except Exception:
        return float(default)

    thr = np.asarray(thr, dtype=float)
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    finite = np.isfinite(thr)
    if not np.any(finite):
        return float(default)

    thr_f = thr[finite]
    j = (tpr - fpr)[finite]

    best_j = np.max(j)
    cand = thr_f[j == best_j]
    if cand.size == 0:
        return float(default)

    # tie-breaker: closest to default
    best_thr = float(cand[np.argmin(np.abs(cand - float(default)))])
    # clip to probability range
    best_thr = float(np.clip(best_thr, 0.0, 1.0))
    return best_thr


def binary_metrics(y_true: np.ndarray, prob1: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob1 = np.asarray(prob1).astype(float)
    y_pred = (prob1 >= float(threshold)).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    sens = safe_div(tp, tp + fn)  # sensitivity / recall
    spec = safe_div(tn, tn + fp)  # specificity
    ppv = safe_div(tp, tp + fp)   # positive predictive value / precision
    npv = safe_div(tn, tn + fn)   # negative predictive value
    acc = float(accuracy_score(y_true, y_pred))

    try:
        auc = float(roc_auc_score(y_true, prob1)) if len(np.unique(y_true)) == 2 else float("nan")
    except Exception:
        auc = float("nan")

    bacc = float(balanced_accuracy_score(y_true, y_pred))

    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(acc),
        "auc": float(auc),
        "balanced_accuracy": float(bacc),
    }
