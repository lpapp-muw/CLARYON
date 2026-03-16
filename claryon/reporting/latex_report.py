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
{{ row.model }} {% for m in metrics %}& {{ row[m] | round(4) }} {% endfor %}\\
{% endfor %}
\hline
\end{tabular}
\end{table}
"""


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
) -> Path:
    """Generate results table .tex file."""
    return render_latex_report(
        RESULTS_TEMPLATE,
        {"metrics": metrics, "results": results},
        output_path,
    )
