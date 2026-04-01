"""Inference on new data using saved models."""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .io.base import TaskType
from .io.predictions import write_predictions
from .preprocessing.state import PreprocessingState

logger = logging.getLogger(__name__)


def _import_model_modules() -> None:
    """Import model modules to trigger @register decorators."""
    modules = [
        "claryon.models.classical.mlp_",
    ]
    optional = [
        "claryon.models.classical.xgboost_",
        "claryon.models.classical.lightgbm_",
        "claryon.models.classical.catboost_",
        "claryon.models.classical.tabpfn_",
        "claryon.models.classical.tabm_",
        "claryon.models.classical.realmlp_",
        "claryon.models.classical.modernnca_",
        "claryon.models.classical.cnn_2d",
        "claryon.models.classical.cnn_3d",
        "claryon.models.quantum.kernel_svm",
        "claryon.models.quantum.projected_kernel_svm",
        "claryon.models.quantum.angle_pqk_svm",
        "claryon.models.quantum.qcnn_muw",
        "claryon.models.quantum.qcnn_alt",
        "claryon.models.quantum.qdc_hadamard",
        "claryon.models.quantum.quantum_gp",
        "claryon.models.quantum.qnn",
    ]
    for mod in modules:
        importlib.import_module(mod)
    for mod in optional:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass


def run_inference(
    model_dir: str,
    input_path: str,
    output_path: str,
    sep: str = ";",
    id_col: str = "Key",
    label_col: Optional[str] = None,
) -> Path:
    """Run inference on new data using a saved model.

    Args:
        model_dir: Directory containing saved model + preprocessing state.
        input_path: Path to input CSV with new data.
        output_path: Path to write predictions.
        sep: CSV separator for input.
        id_col: ID column name.
        label_col: Optional label column (if present, include in output).

    Returns:
        Path to output predictions file.
    """
    model_dir_path = Path(model_dir)
    output = Path(output_path)

    _import_model_modules()

    # Load model params to determine model name/type
    params_path = model_dir_path / "model_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"model_params.json not found in {model_dir}")
    with open(params_path) as f:
        params = json.load(f)

    # Detect model name from directory structure (model_name/seed_X/fold_Y/)
    # Walk up to find model name
    model_name = _detect_model_name(model_dir_path)

    # Load preprocessing state
    preproc_path = model_dir_path / "preprocessing_state.json"
    preproc_state = None
    if preproc_path.exists():
        preproc_state = PreprocessingState.load(preproc_path)
        logger.info("Loaded preprocessing state: %d → %d features",
                     preproc_state.n_features_original,
                     preproc_state.n_features_selected)

    # Load model
    from .registry import get
    model_cls = get("model", model_name)
    model = model_cls(**params)
    model.load(model_dir_path)
    logger.info("Loaded model: %s from %s", model_name, model_dir_path)

    # Load new data
    df = pd.read_csv(input_path, sep=sep)
    keys = df[id_col].tolist() if id_col in df.columns else [f"S{i:04d}" for i in range(len(df))]

    # Extract labels if present
    y_true = None
    if label_col and label_col in df.columns:
        y_true = df[label_col].to_numpy()
        df = df.drop(columns=[label_col])

    # Drop ID column
    if id_col in df.columns:
        df = df.drop(columns=[id_col])

    # Auto-detect label column if not explicitly specified and column count
    # exceeds what the preprocessing state expects
    if preproc_state is not None and y_true is None:
        expected = preproc_state.n_features_original
        if len(df.columns) > expected:
            _label_candidates = {"label", "target", "class", "y", "Label", "Target", "Class"}
            for col in list(df.columns):
                if col in _label_candidates:
                    logger.info("Auto-detected label column '%s' (expected %d features, got %d)",
                                col, expected, len(df.columns))
                    y_true = df[col].to_numpy()
                    df = df.drop(columns=[col])
                    break

    X = df.to_numpy(dtype=np.float64)

    # Apply preprocessing
    if preproc_state is not None:
        X = preproc_state.apply_tabular(X)

    # Amplitude encoding for quantum models
    n_qubits = params.get("n_qubits")
    if n_qubits is not None:
        from .encoding.amplitude import amplitude_encode_matrix
        pad_len = 2 ** n_qubits
        X, _ = amplitude_encode_matrix(X, pad_len=pad_len)

    # Predict
    probs = model.predict_proba(X)
    predicted = np.argmax(probs, axis=1)

    # Write output
    write_predictions(
        output,
        keys=keys,
        actual=y_true,
        predicted=predicted,
        probabilities=probs,
        task_type=TaskType.BINARY if probs.shape[1] == 2 else TaskType.MULTICLASS,
    )
    logger.info("Predictions written to %s (%d samples)", output, len(keys))
    return output


def _detect_model_name(model_dir: Path) -> str:
    """Detect model name from directory structure.

    Expects: Results/<experiment>/<model_name>/seed_X/fold_Y/

    Args:
        model_dir: The fold-level directory.

    Returns:
        Model name string.
    """
    # Walk up: fold_Y -> seed_X -> model_name
    parent = model_dir
    for _ in range(3):
        if parent.name.startswith("seed_") or parent.name.startswith("fold_"):
            parent = parent.parent
        else:
            break
    return parent.name
