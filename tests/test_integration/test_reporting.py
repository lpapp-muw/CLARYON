"""Integration tests for reporting — LaTeX and Markdown generation."""
from __future__ import annotations

import pytest


def test_generate_latex_methods(tmp_path):
    from claryon.reporting.latex_report import generate_methods_section

    path = generate_methods_section(
        experiment_name="test_exp",
        seed=42,
        cv_strategy="kfold",
        n_folds=5,
        models=["xgboost", "mlp"],
        metrics=["bacc", "auc"],
        output_path=tmp_path / "methods.tex",
    )
    assert path.exists()
    content = path.read_text()
    assert "test_exp" in content
    assert "xgboost" in content
    assert "bacc" in content


def test_generate_latex_results(tmp_path):
    from claryon.reporting.latex_report import generate_results_section

    results = [
        {"model": "xgboost", "bacc": 0.85, "auc": 0.92},
        {"model": "mlp", "bacc": 0.80, "auc": 0.88},
    ]
    path = generate_results_section(
        metrics=["bacc", "auc"],
        results=results,
        output_path=tmp_path / "results.tex",
    )
    assert path.exists()
    content = path.read_text()
    assert "xgboost" in content
    assert "0.85" in content


def test_generate_markdown_report(tmp_path):
    from claryon.reporting.markdown_report import generate_markdown_report

    results = [
        {"model": "xgboost", "bacc": 0.85, "auc": 0.92},
        {"model": "mlp", "bacc": 0.80, "auc": 0.88},
    ]
    path = generate_markdown_report(
        experiment_name="test_exp",
        seed=42,
        cv_strategy="kfold",
        n_folds=5,
        models=["xgboost", "mlp"],
        metrics=["bacc", "auc"],
        results=results,
        output_path=tmp_path / "report.md",
    )
    assert path.exists()
    content = path.read_text()
    assert "# test_exp" in content
    assert "0.8500" in content
    assert "| xgboost" in content
