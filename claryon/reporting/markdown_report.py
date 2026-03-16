"""Markdown report generator — Jinja2 → .md files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import jinja2

logger = logging.getLogger(__name__)

MD_TEMPLATE = """# {{ experiment_name }} — Results Report

## Setup
- **Seed**: {{ seed }}
- **CV Strategy**: {{ cv_strategy }}, {{ n_folds }} folds
- **Models**: {{ models | join(', ') }}
- **Metrics**: {{ metrics | join(', ') }}

## Results

| Model {% for m in metrics %}| {{ m }} {% endfor %}|
|---{% for m in metrics %}|---{% endfor %}|
{% for row in results %}| {{ row.model }} {% for m in metrics %}| {{ "%.4f" | format(row[m]) }} {% endfor %}|
{% endfor %}
"""


def generate_markdown_report(
    experiment_name: str,
    seed: int,
    cv_strategy: str,
    n_folds: int,
    models: list[str],
    metrics: list[str],
    results: list[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate a Markdown results report.

    Args:
        experiment_name: Experiment name.
        seed: Random seed.
        cv_strategy: CV strategy name.
        n_folds: Number of folds.
        models: List of model names.
        metrics: List of metric names.
        results: List of dicts with model + metric values.
        output_path: Output .md file path.

    Returns:
        Path to written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        undefined=jinja2.StrictUndefined,
    )
    template = env.from_string(MD_TEMPLATE)
    rendered = template.render(
        experiment_name=experiment_name,
        seed=seed,
        cv_strategy=cv_strategy,
        n_folds=n_folds,
        models=models,
        metrics=metrics,
        results=results,
    )

    output_path.write_text(rendered)
    logger.info("Wrote Markdown report to %s", output_path)
    return output_path
