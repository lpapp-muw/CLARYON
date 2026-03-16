"""Integration test: classical pipeline end-to-end.

Config → load → split → train (MLP) → predict → write predictions.
Uses synthetic tabular binary fixture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from claryon.config_schema import load_config
from claryon.io.predictions import read_predictions
from claryon.pipeline import run_pipeline


def test_classical_pipeline_binary(tabular_binary_dir, tmp_path):
    """Full pipeline: load tabular binary → 2-fold CV → MLP → predictions."""
    # Write config YAML
    config_data = {
        "experiment": {
            "name": "test_binary",
            "seed": 42,
            "results_dir": str(tmp_path / "results"),
        },
        "data": {
            "tabular": {
                "path": str(tabular_binary_dir / "train.csv"),
                "label_col": "label",

                "sep": ";",
            },
        },
        "cv": {
            "strategy": "kfold",
            "n_folds": 2,
            "seeds": [42],
        },
        "models": [
            {
                "name": "mlp",
                "type": "tabular",
                "params": {
                    "hidden_layer_sizes": [16],
                    "max_iter": 50,
                    "random_state": 42,
                },
            },
        ],
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))
    config = load_config(config_path)

    state = run_pipeline(config)

    # Verify dataset loaded
    assert state.dataset is not None
    assert state.dataset.n_samples == 80

    # Verify splits generated
    assert 42 in state.splits
    assert len(state.splits[42]) == 2

    # Verify model results
    assert "mlp" in state.results
    assert len(state.results["mlp"]) == 2  # 2 folds
    assert all(r["status"] == "ok" for r in state.results["mlp"])

    # Verify prediction files written
    results_dir = tmp_path / "results"
    for fold in range(2):
        pred_path = results_dir / "mlp" / "seed_42" / f"fold_{fold}" / "Predictions.csv"
        assert pred_path.exists(), f"Missing predictions for fold {fold}"
        df = read_predictions(pred_path)
        assert len(df) > 0
        assert "Key" in df.columns
        assert "Actual" in df.columns
        assert "Predicted" in df.columns
        assert "P0" in df.columns
        assert "P1" in df.columns


def test_classical_pipeline_multiclass(tabular_multiclass_dir, tmp_path):
    """Pipeline with multiclass data."""
    config_data = {
        "experiment": {
            "name": "test_multiclass",
            "seed": 42,
            "results_dir": str(tmp_path / "results"),
        },
        "data": {
            "tabular": {
                "path": str(tabular_multiclass_dir / "train.csv"),
                "label_col": "label",

                "sep": ";",
            },
        },
        "cv": {
            "strategy": "kfold",
            "n_folds": 2,
            "seeds": [42],
        },
        "models": [
            {
                "name": "mlp",
                "type": "tabular",
                "params": {
                    "hidden_layer_sizes": [16],
                    "max_iter": 50,
                    "random_state": 42,
                },
            },
        ],
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))
    config = load_config(config_path)

    state = run_pipeline(config)

    assert state.dataset.task_type.value == "multiclass"
    assert "mlp" in state.results
    assert all(r["status"] == "ok" for r in state.results["mlp"])

    # Check 3-class probabilities in output
    pred_path = tmp_path / "results" / "mlp" / "seed_42" / "fold_0" / "Predictions.csv"
    df = read_predictions(pred_path)
    assert "P0" in df.columns
    assert "P1" in df.columns
    assert "P2" in df.columns
