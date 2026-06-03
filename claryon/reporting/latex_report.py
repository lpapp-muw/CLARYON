"""LaTeX report generator — Jinja2 → .tex files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import jinja2

logger = logging.getLogger(__name__)

METHODS_TEMPLATE = r"""
\section{Methods}
\subsection{Experimental Setup}
Experiment: {{ experiment_name }}.
Seed: {{ seed }}. CV strategy: {{ cv_strategy }}, {{ n_folds }} folds.

\subsection{Models}
{% for model in models %}
\textbf{{ '{' }}{{ model }}{{ '}' }}{% if not loop.last %}, {% endif %}
{% endfor %}

\subsection{Metrics}
{% for metric in metrics %}
{{ metric }}{% if not loop.last %}, {% endif %}
{% endfor %}
"""

RESULTS_TEMPLATE = r"""
\section{Results}

\begin{table}[ht]
\centering
\caption{Model performance summary}
\begin{tabular}{{ '{' }}l{% for m in metrics %}r{% endfor %}{{ '}' }}
\hline
Model {% for m in metrics %}& {{ m }} {% endfor %}\\
\hline
{% for row in results %}
{{ row.model }} {% for m in metrics %}& {{ format_metric(row, m) }} {% endfor %}\\
{% endfor %}
\hline
\end{tabular}
\end{table}
"""


def _format_metric_latex(row: dict, metric: str) -> str:
    """Format a metric value as 'mean $\\pm$ std' for LaTeX."""
    import math
    val = row.get(metric, float("nan"))
    std_key = f"{metric}_std"
    std = row.get(std_key)
    try:
        if math.isnan(val):
            return "NaN"
    except (TypeError, ValueError):
        return str(val)
    if std is not None:
        try:
            if not math.isnan(std):
                return f"{val:.4f} $\\pm$ {std:.4f}"
        except (TypeError, ValueError):
            pass
    return f"{val:.4f}"


def render_latex_report(
    template_str: str,
    context: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Render a Jinja2 LaTeX template to file.

    Args:
        template_str: Jinja2 template string.
        context: Template variables.
        output_path: Output .tex file path.

    Returns:
        Path to written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        undefined=jinja2.StrictUndefined,
    )
    env.globals["format_metric"] = _format_metric_latex
    template = env.from_string(template_str)
    rendered = template.render(**context)

    output_path.write_text(rendered)
    logger.info("Wrote LaTeX report to %s", output_path)
    return output_path


def generate_methods_section(
    experiment_name: str,
    seed: int,
    cv_strategy: str,
    n_folds: int,
    models: list[str],
    metrics: list[str],
    output_path: Path,
) -> Path:
    """Generate methods section .tex file."""
    return render_latex_report(
        METHODS_TEMPLATE,
        {
            "experiment_name": experiment_name,
            "seed": seed,
            "cv_strategy": cv_strategy,
            "n_folds": n_folds,
            "models": models,
            "metrics": metrics,
        },
        output_path,
    )


def generate_results_section(
    metrics: list[str],
    results: list[Dict[str, Any]],
    output_path: Path,
    include_ensemble: bool = True,
) -> Path:
    """Generate results table .tex file.

    If there are multiple models and include_ensemble is True, an Ensemble
    row is appended showing the mean across models for each metric.
    """
    rows = list(results)
    if include_ensemble and len(rows) > 1:
        ensemble_row: Dict[str, Any] = {"model": "Ensemble"}
        for m in metrics:
            vals = [r[m] for r in rows if m in r and not _is_nan(r[m])]
            ensemble_row[m] = sum(vals) / len(vals) if vals else float("nan")
        rows.append(ensemble_row)
    return render_latex_report(
        RESULTS_TEMPLATE,
        {"metrics": metrics, "results": rows},
        output_path,
    )


def _is_nan(v: Any) -> bool:
    """Check if a value is NaN."""
    try:
        import math
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False
