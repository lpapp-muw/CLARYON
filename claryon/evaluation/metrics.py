"""Metric registry and implementations — binary, multiclass, regression metrics.

Ported from [E] metrics.py + extended. All metrics registered via @register.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)

from ..registry import register

logger = logging.getLogger(__name__)


def safe_div(a: float, b: float) -> float:
    """Safe division returning NaN on zero denominator."""
    return float(a / b) if b != 0 else float("nan")


def _unique_labels(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Sorted union of the observed true and predicted labels."""
    return sorted(set(np.unique(y_true)).union(np.unique(y_pred)))


def _ovr_counts(cm: np.ndarray, c: int) -> tuple:
    """One-vs-rest (tp, fn, fp, tn) for class index ``c`` from a K x K CM."""
    total = int(cm.sum())
    tp = int(cm[c, c])
    fn = int(cm[c, :].sum()) - tp
    fp = int(cm[:, c].sum()) - tp
    tn = total - tp - fn - fp
    return tp, fn, fp, tn


def _macro_ovr(y_true: np.ndarray, y_pred: np.ndarray, ratio) -> float:
    """Macro-average a per-class one-vs-rest ratio over all observed classes.

    For 2 classes this reproduces the original binary definition on labels
    ``[0, 1]`` (one-vs-rest on the positive class 1 is algebraically identical).
    For K > 2 it macro-averages the per-class one-vs-rest ratio, skipping classes
    whose denominator is zero (NaN-safe).

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        ratio: Callable ``(tp, fn, fp, tn) -> float`` producing the metric value.

    Returns:
        The (binary or macro-averaged) metric value, or NaN when undefined.
    """
    labels = _unique_labels(y_true, y_pred)
    if len(labels) <= 2:
        labels = [0, 1]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return ratio(*_ovr_counts(cm, 1))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    vals = [ratio(*_ovr_counts(cm, c)) for c in range(len(labels))]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def select_threshold_balanced_accuracy(
    y_true: np.ndarray,
    prob1: np.ndarray,
    default: float = 0.5,
) -> float:
    """Select decision threshold via Youden's J on training data.

    Args:
        y_true: Binary labels (0/1).
        prob1: Predicted P(class=1).
        default: Fallback threshold.

    Returns:
        Optimal threshold in [0, 1].
    """
    y_true = np.asarray(y_true).astype(int)
    prob1 = np.asarray(prob1).astype(float)

    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float(default)

    try:
        fpr, tpr, thr = roc_curve(y_true, prob1)
    except Exception:
        return float(default)

    finite = np.isfinite(thr)
    if not np.any(finite):
        return float(default)

    j = (tpr - fpr)[finite]
    thr_f = thr[finite]
    best_j = np.max(j)
    cand = thr_f[j == best_j]
    if cand.size == 0:
        return float(default)

    best_thr = float(cand[np.argmin(np.abs(cand - float(default)))])
    return float(np.clip(best_thr, 0.0, 1.0))


@register("metric", "bacc")
def metric_bacc(y_true: np.ndarray, y_pred: np.ndarray, **kw: object) -> float:
    """Balanced accuracy."""
    return float(balanced_accuracy_score(y_true, y_pred))


@register("metric", "accuracy")
def metric_accuracy(y_true: np.ndarray, y_pred: np.ndarray, **kw: object) -> float:
    """Accuracy."""
    return float(accuracy_score(y_true, y_pred))


@register("metric", "sensitivity")
def metric_sensitivity(y_true: np.ndarray, y_pred: np.ndarray, **kw: object) -> float:
    """Sensitivity (recall, TPR). Binary on labels [0, 1]; macro one-vs-rest for K > 2."""
    return _macro_ovr(y_true, y_pred, lambda tp, fn, fp, tn: safe_div(tp, tp + fn))


@register("metric", "specificity")
def metric_specificity(y_true: np.ndarray, y_pred: np.ndarray, **kw: object) -> float:
    """Specificity (TNR). Binary on labels [0, 1]; macro one-vs-rest for K > 2."""
    return _macro_ovr(y_true, y_pred, lambda tp, fn, fp, tn: safe_div(tn, tn + fp))


@register("metric", "ppv")
def metric_ppv(y_true: np.ndarray, y_pred: np.ndarray, **kw: object) -> float:
    """Positive predictive value (precision). Binary on [0, 1]; macro one-vs-rest for K > 2."""
    return _macro_ovr(y_true, y_pred, lambda tp, fn, fp, tn: safe_div(tp, tp + fp))


@register("metric", "npv")
def metric_npv(y_true: np.ndarray, y_pred: np.ndarray, **kw: object) -> float:
    """Negative predictive value. Binary on [0, 1]; macro one-vs-rest for K > 2."""
    return _macro_ovr(y_true, y_pred, lambda tp, fn, fp, tn: safe_div(tn, tn + fn))


@register("metric", "auc")
def metric_auc(y_true: np.ndarray, y_pred: np.ndarray, probabilities: Optional[np.ndarray] = None, **kw: object) -> float:
    """Area under ROC curve. Requires probabilities."""
    if probabilities is None:
        return float("nan")
    try:
        n_classes = probabilities.shape[1] if probabilities.ndim > 1 else 2
        if n_classes > 2:
            return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted"))
        else:
            probs = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
            return float(roc_auc_score(y_true, probs))
    except Exception as e:
        logger.warning("AUC computation failed: %s", e)
        return float("nan")


@register("metric", "logloss")
def metric_logloss(y_true: np.ndarray, y_pred: np.ndarray, probabilities: Optional[np.ndarray] = None, **kw: object) -> float:
    """Log loss / cross-entropy. Requires probabilities."""
    if probabilities is None:
        return float("nan")
    try:
        probs = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
        if probs.ndim == 2:
            probs = probs / probs.sum(axis=1, keepdims=True)
        return float(log_loss(y_true, probs))
    except Exception:
        return float("nan")


@register("metric", "mse")
def metric_mse(y_true: np.ndarray, y_pred: np.ndarray, **kw: object) -> float:
    """Mean squared error (regression)."""
    return float(mean_squared_error(y_true, y_pred))


@register("metric", "mae")
def metric_mae(y_true: np.ndarray, y_pred: np.ndarray, **kw: object) -> float:
    """Mean absolute error (regression)."""
    return float(mean_absolute_error(y_true, y_pred))


@register("metric", "r2")
def metric_r2(y_true: np.ndarray, y_pred: np.ndarray, **kw: object) -> float:
    """R-squared (regression)."""
    return float(r2_score(y_true, y_pred))


def binary_metrics(y_true: np.ndarray, prob1: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute all binary classification metrics.

    Args:
        y_true: Binary labels (0/1).
        prob1: Predicted P(class=1).
        threshold: Decision threshold.

    Returns:
        Dict of metric name → value.
    """
    y_pred = (prob1 >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    try:
        auc = float(roc_auc_score(y_true, prob1))
    except Exception:
        auc = float("nan")

    return {
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "ppv": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
