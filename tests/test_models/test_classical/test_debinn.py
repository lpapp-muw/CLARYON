"""Tests for claryon.models.classical.debinn_ — DEBI-NN mock binary test."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from claryon.io.base import TaskType
from claryon.models.classical.debinn_ import DEBINNModel


MOCK_BINARY = Path(__file__).resolve().parents[2] / "fixtures" / "mock_debinn.sh"


def test_debinn_registered():
    from claryon.registry import get
    # Need to import to trigger registration
    import claryon.models.classical.debinn_  # noqa: F401
    cls = get("model", "debinn")
    assert cls is DEBINNModel


def test_debinn_mock_binary(tmp_path):
    """Test DEBI-NN with mock binary."""
    if not MOCK_BINARY.exists():
        pytest.skip("Mock DEBI-NN binary not found")

    # Create a minimal project directory
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    model = DEBINNModel(binary_path=str(MOCK_BINARY), timeout=30)
    model.fit(
        np.zeros((5, 10)), np.array([0, 1, 0, 1, 0]),
        TaskType.BINARY, project_dir=str(project_dir),
    )

    # Load the predictions the mock wrote
    pred_path = (
        project_dir / "Executions-Finished" / "test_dataset-M0" / "Log" / "Fold-1" / "Predictions.csv"
    )
    assert pred_path.exists()
    model.load_predictions(pred_path)

    preds = model.predict(np.zeros((5, 10)))
    assert len(preds) == 5

    probs = model.predict_proba(np.zeros((5, 10)))
    assert probs.shape == (5, 2)


def test_debinn_no_binary():
    model = DEBINNModel(binary_path="/nonexistent/debinn")
    with pytest.raises(RuntimeError, match="DEBI-NN failed"):
        model.fit(
            np.zeros((5, 10)), np.array([0, 1, 0, 1, 0]),
            TaskType.BINARY, project_dir="/tmp/fake",
        )


def test_debinn_no_project_dir():
    model = DEBINNModel()
    with pytest.raises(ValueError, match="project_dir"):
        model.fit(np.zeros((1, 1)), np.array([0]), TaskType.BINARY)
