"""Tests for claryon.evaluation.figures — figure generation."""
from __future__ import annotations

import numpy as np
import pytest


def test_plot_roc_curve(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from claryon.evaluation.figures import plot_roc_curve

    y_true = np.array([0, 0, 1, 1, 0, 1])
    prob1 = np.array([0.1, 0.3, 0.8, 0.9, 0.4, 0.7])
    save_path = tmp_path / "roc.png"
    plot_roc_curve(y_true, prob1, save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_confusion_matrix(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from claryon.evaluation.figures import plot_confusion_matrix

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    save_path = tmp_path / "cm.png"
    plot_confusion_matrix(y_true, y_pred, save_path=save_path)
    assert save_path.exists()


def test_plot_cd_diagram(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from claryon.evaluation.figures import plot_critical_difference_diagram

    mean_ranks = {"XGBoost": 1.5, "LightGBM": 2.0, "MLP": 3.0}
    save_path = tmp_path / "cd.png"
    plot_critical_difference_diagram(mean_ranks, cd=0.8, save_path=save_path)
    assert save_path.exists()
