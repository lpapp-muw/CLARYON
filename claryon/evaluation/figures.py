"""Publication-quality figure generators — ROC, confusion matrix, CD diagram.

New module + partially ported from [B] analysis.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def plot_roc_curve(
    y_true: np.ndarray,
    prob1: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """Plot ROC curve with AUC.

    Args:
        y_true: Binary labels.
        prob1: P(class=1) predictions.
        title: Plot title.
        save_path: If given, save figure to this path.
        dpi: Figure DPI.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(y_true, prob1)
    auc = roc_auc_score(y_true, prob1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """Plot confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Class label names.
        title: Plot title.
        save_path: If given, save figure.
        dpi: Figure DPI.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_critical_difference_diagram(
    mean_ranks: Dict[str, float],
    cd: float,
    title: str = "Critical Difference Diagram",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """Plot critical difference diagram for Friedman/Nemenyi test.

    Args:
        mean_ranks: Dict of method_name → mean rank.
        cd: Critical difference value.
        title: Plot title.
        save_path: If given, save figure.
        dpi: Figure DPI.
    """
    import matplotlib.pyplot as plt

    sorted_methods = sorted(mean_ranks.items(), key=lambda x: x[1])
    names = [m for m, _ in sorted_methods]
    ranks = [r for _, r in sorted_methods]
    k = len(names)

    fig, ax = plt.subplots(1, 1, figsize=(max(8, k * 1.2), 3))
    ax.set_xlim(0.5, k + 0.5)

    for i, (name, rank) in enumerate(zip(names, ranks)):
        ax.plot(rank, 0.5, "ko", markersize=8)
        ax.annotate(f"{name}\n({rank:.2f})", (rank, 0.5),
                    textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=9)

    # CD bar
    ax.plot([1, 1 + cd], [0.8, 0.8], "r-", lw=3)
    ax.annotate(f"CD = {cd:.2f}", (1 + cd / 2, 0.8),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color="red")

    ax.set_xlabel("Mean Rank")
    ax.set_title(title)
    ax.set_yticks([])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
