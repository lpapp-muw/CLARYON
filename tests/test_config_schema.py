"""Tests for claryon.config_schema — Pydantic config validation."""
from __future__ import annotations

import pytest

from claryon.config_schema import ClaryonConfig, load_config


VALID_YAML = """\
experiment:
  name: test_experiment
  seed: 42
  results_dir: Results

data:
  tabular:
    path: data/train.csv
    label_col: label
    id_col: Key

cv:
  strategy: kfold
  n_folds: 5
  seeds: [42, 123]

models:
  - name: xgboost
    type: tabular
  - name: pl_qcnn_alt
    type: tabular_quantum
    params:
      epochs: 15
      lr: 0.02

explainability:
  shap: true
  lime: true

evaluation:
  metrics: [bacc, auc]

reporting:
  markdown: true
"""


def test_valid_yaml_parses(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(VALID_YAML)
    config = load_config(cfg_path)
    assert config.experiment.name == "test_experiment"
    assert config.experiment.seed == 42
    assert len(config.models) == 2
    assert config.models[0].name == "xgboost"
    assert config.cv.n_folds == 5


def test_defaults_filled():
    config = ClaryonConfig()
    assert config.experiment.seed == 42
    assert config.cv.strategy == "kfold"
    assert config.reporting.figure_dpi == 300


def test_invalid_cv_folds_rejected():
    with pytest.raises(Exception):
        ClaryonConfig(cv={"strategy": "kfold", "n_folds": 1})


def test_invalid_strategy_rejected():
    with pytest.raises(Exception):
        ClaryonConfig(cv={"strategy": "invalid_strategy"})


def test_disabled_models_filtered():
    config = ClaryonConfig(
        models=[
            {"name": "a", "type": "tabular", "enabled": True},
            {"name": "b", "type": "tabular", "enabled": False},
        ]
    )
    assert len(config.models) == 1
    assert config.models[0].name == "a"


def test_empty_yaml(tmp_path):
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("")
    config = load_config(cfg_path)
    assert config.experiment.name == "experiment"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")
