"""Full integration test — exercises the complete pipeline from config to report.

This is the Phase 7 capstone test: load → split → train → predict → evaluate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from claryon.config_schema import load_config
from claryon.evaluation.metrics import metric_bacc, metric_auc
from claryon.io.predictions import read_predictions
from claryon.pipeline import run_pipeline


def test_full_pipeline_binary_with_evaluation(tabular_binary_dir, tmp_path):
    """Full pipeline: tabular binary → MLP → predictions → metrics check."""
    config_data = {
        "experiment": {
            "name": "full_integration_binary",
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
            "n_folds": 3,
            "seeds": [42, 123],
        },
        "models": [
            {
                "name": "mlp",
                "type": "tabular",
                "params": {"hidden_layer_sizes": [16], "max_iter": 50, "random_state": 42},
            },
        ],
        "evaluation": {
            "metrics": ["bacc", "auc"],
        },
    }

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config_data))
    config = load_config(cfg_path)
    state = run_pipeline(config)

    # 2 seeds × 3 folds = 6 results
    assert len(state.results["mlp"]) == 6
    assert all(r["status"] == "ok" for r in state.results["mlp"])

    # Check all prediction files exist and can compute metrics
    results_dir = tmp_path / "results"
    for seed in [42, 123]:
        for fold in range(3):
            pred_path = results_dir / "mlp" / f"seed_{seed}" / f"fold_{fold}" / "Predictions.csv"
            assert pred_path.exists()
            df = read_predictions(pred_path)
            y_true = df["Actual"].values
            y_pred = df["Predicted"].values
            probs = df[["P0", "P1"]].values
            bacc = metric_bacc(y_true, y_pred)
            auc = metric_auc(y_true, y_pred, probabilities=probs)
            # Should be non-NaN
            assert not np.isnan(bacc)
            assert not np.isnan(auc)


def test_full_pipeline_multimodel(tabular_binary_dir, tmp_path):
    """Full pipeline with multiple models — verify all produce predictions."""
    config_data = {
        "experiment": {
            "name": "multimodel_test",
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
            "strategy": "holdout",
            "n_folds": 5,
            "seeds": [42],
            "test_size": 0.2,
        },
        "models": [
            {"name": "mlp", "type": "tabular", "params": {"hidden_layer_sizes": [8], "max_iter": 30}},
            {"name": "xgboost", "type": "tabular", "params": {"n_estimators": 10}},
        ],
    }

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config_data))
    config = load_config(cfg_path)
    state = run_pipeline(config)

    assert "mlp" in state.results
    assert "xgboost" in state.results
    assert len(state.results["mlp"]) == 1  # holdout = 1 split
    assert len(state.results["xgboost"]) == 1
