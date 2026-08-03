"""Confusion-matrix artifacts and correct multiclass sensitivity/specificity.

The registered ``sensitivity``/``specificity`` metrics collapse to a binary
2x2 slice (``labels=[0, 1]``), which is invalid for K > 2 classes. This module
reconstructs the full K x K confusion matrix from the per-fold ``Predictions.csv``
files that the pipeline already writes, and derives one-vs-rest per-class and
macro-averaged sensitivity/specificity that are correct for any number of classes.

It is invoked automatically from :func:`claryon.pipeline.stage_evaluate` (writing
``<model>/confusion_matrix.csv`` and a combined ``confusion_report.json`` under the
results directory), and can also be run standalone::

    python -m claryon.evaluation.confusion <results_dir> [model]
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEP = ";"


def _per_class_sens_spec(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One-vs-rest sensitivity (recall) and specificity per class from a K x K CM.

    Args:
        cm: Confusion matrix, shape (K, K), rows = actual, cols = predicted.

    Returns:
        Tuple ``(sensitivity, specificity)``, each shape (K,). Entries are NaN
        where the relevant denominator is zero (no support for that class).
    """
    k = cm.shape[0]
    total = int(cm.sum())
    sens = np.full(k, np.nan)
    spec = np.full(k, np.nan)
    for c in range(k):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum()) - tp
        fp = int(cm[:, c].sum()) - tp
        tn = total - tp - fn - fp
        sens[c] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec[c] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sens, spec


def _pred_files(results_dir: Union[str, Path], model: str) -> List[str]:
    """List all ``Predictions.csv`` paths for a model across seeds and folds."""
    pat = os.path.join(str(results_dir), model, "seed_*", "fold_*", "Predictions.csv")
    return sorted(glob.glob(pat))


def _discover_models(results_dir: Union[str, Path]) -> List[str]:
    """Find model subdirectories under ``results_dir`` that hold prediction files."""
    pat = os.path.join(str(results_dir), "*", "seed_*", "fold_*", "Predictions.csv")
    models = {os.path.relpath(f, str(results_dir)).split(os.sep)[0] for f in glob.glob(pat)}
    return sorted(models)


def compute_confusion(results_dir: Union[str, Path], model: str) -> Optional[Dict[str, object]]:
    """Pool a model's predictions and compute its confusion matrix and metrics.

    Args:
        results_dir: Experiment results directory.
        model: Model name (subdirectory under ``results_dir``).

    Returns:
        A dict with keys ``labels``, ``matrix`` (list of lists), ``per_class_sensitivity``,
        ``per_class_specificity``, ``macro_sensitivity``, ``macro_specificity``, ``n_samples``
        and ``n_fold_files``. Returns ``None`` if no prediction files are found.
    """
    from sklearn.metrics import confusion_matrix

    files = _pred_files(results_dir, model)
    if not files:
        return None

    df = pd.concat([pd.read_csv(f, sep=SEP) for f in files], ignore_index=True)
    y_true = df["Actual"].to_numpy()
    y_pred = df["Predicted"].to_numpy()
    labels = sorted(set(np.unique(y_true)).union(np.unique(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sens, spec = _per_class_sens_spec(cm)

    return {
        "labels": [int(x) for x in labels],
        "matrix": cm.astype(int).tolist(),
        "per_class_sensitivity": {int(labels[i]): _nan_to_none(sens[i]) for i in range(len(labels))},
        "per_class_specificity": {int(labels[i]): _nan_to_none(spec[i]) for i in range(len(labels))},
        "macro_sensitivity": _nan_to_none(np.nanmean(sens)) if np.any(np.isfinite(sens)) else None,
        "macro_specificity": _nan_to_none(np.nanmean(spec)) if np.any(np.isfinite(spec)) else None,
        "n_samples": int(len(df)),
        "n_fold_files": len(files),
    }


def _nan_to_none(x: float) -> Optional[float]:
    """Convert a NaN float to ``None`` (JSON-safe), otherwise round to 6 places."""
    xf = float(x)
    return None if not np.isfinite(xf) else round(xf, 6)


def _write_matrix_csv(path: Path, labels: List[int], matrix: List[List[int]]) -> None:
    """Write a labelled K x K confusion matrix as a semicolon-separated CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [""] + [f"pred_{c}" for c in labels]
    lines = [SEP.join(header)]
    for i, c in enumerate(labels):
        lines.append(SEP.join([f"true_{c}"] + [str(int(v)) for v in matrix[i]]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confusion_reports(
    results_dir: Union[str, Path],
    models: Optional[List[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Compute and persist confusion matrices for all models in an experiment.

    For each model this writes ``<results_dir>/<model>/confusion_matrix.csv`` and
    aggregates every model into ``<results_dir>/confusion_report.json``.

    Args:
        results_dir: Experiment results directory.
        models: Explicit model names; if ``None``, discovered from prediction files.

    Returns:
        Mapping of model name to its confusion report dict (see :func:`compute_confusion`).
    """
    results_dir = Path(results_dir)
    if models is None:
        models = _discover_models(results_dir)

    report: Dict[str, Dict[str, object]] = {}
    for model in models:
        info = compute_confusion(results_dir, model)
        if info is None:
            logger.warning("No predictions found for model '%s' — skipping confusion matrix", model)
            continue
        _write_matrix_csv(
            results_dir / model / "confusion_matrix.csv",
            info["labels"],  # type: ignore[arg-type]
            info["matrix"],  # type: ignore[arg-type]
        )
        report[model] = info

    if report:
        out = results_dir / "confusion_report.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote confusion report for %d model(s) to %s", len(report), out)
    return report


def _print_report(results_dir: str, model: Optional[str] = None) -> None:
    """Pretty-print confusion matrices and metrics to stdout (standalone use)."""
    models = [model] if model else _discover_models(results_dir)
    if not models:
        print(f"No Predictions.csv found under: {results_dir}")
        return
    for m in models:
        info = compute_confusion(results_dir, m)
        if info is None:
            print(f"[{m}] no predictions")
            continue
        labels = info["labels"]  # type: ignore[assignment]
        matrix = info["matrix"]  # type: ignore[assignment]
        print(f"\n===== {m}  |  {info['n_fold_files']} fold-files pooled, N={info['n_samples']} (all seeds) =====")
        print("rows = Actual, cols = Predicted")
        print("        " + "".join(f"pred{c:>3}" for c in labels))
        for i, c in enumerate(labels):
            print(f"true{c:>3} " + "".join(f"{matrix[i][j]:7d}" for j in range(len(labels))))
        print("per-class sensitivity:", info["per_class_sensitivity"])
        print("per-class specificity:", info["per_class_specificity"])
        print(f"macro sensitivity = {info['macro_sensitivity']}   macro specificity = {info['macro_specificity']}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: ``python -m claryon.evaluation.confusion <results_dir> [model]``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m claryon.evaluation.confusion <results_dir> [model]")
        sys.exit(1)
    _print_report(argv[0], argv[1] if len(argv) > 1 else None)


if __name__ == "__main__":
    main()
