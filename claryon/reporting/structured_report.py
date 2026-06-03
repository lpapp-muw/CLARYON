"""Structured LaTeX methods generator — assembles prose from a text registry.

Reads method_descriptions.yaml, selects blocks based on what was configured,
interpolates parameters, and produces a coherent methods section.

Usage:
    from claryon.reporting.structured_report import generate_structured_methods
    generate_structured_methods(config, state, output_path)
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..config_schema import ClaryonConfig

logger = logging.getLogger(__name__)

# Path to the text registry YAML
_DESCRIPTIONS_PATH = Path(__file__).parent / "method_descriptions.yaml"

# LaTeX escapes
_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "~": r"\textasciitilde{}",
}


def _latex_escape(s: str) -> str:
    """Escape special LaTeX characters in plain text (not in math mode)."""
    for char, replacement in _LATEX_SPECIAL.items():
        s = s.replace(char, replacement)
    return s


def _load_descriptions() -> Dict[str, Any]:
    """Load the method description registry."""
    if not _DESCRIPTIONS_PATH.exists():
        logger.warning("Method descriptions not found: %s", _DESCRIPTIONS_PATH)
        return {}
    with open(_DESCRIPTIONS_PATH) as f:
        return yaml.safe_load(f) or {}


def _get_text(
    registry: Dict[str, Any],
    category: str,
    key: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Look up a text block and interpolate context variables.

    Args:
        registry: The loaded descriptions dict.
        category: Top-level key (e.g., 'models', 'cv', 'metrics').
        key: Item key within the category.
        context: Variables for {{param}} interpolation.

    Returns:
        Interpolated text, or empty string if not found.
    """
    cat = registry.get(category, {})
    entry = cat.get(key, {})
    text = entry.get("text", "")
    if not text:
        return ""
    if context:
        try:
            # Simple {{key}} replacement (not full Jinja2 — avoids LaTeX conflicts)
            for k, v in context.items():
                text = text.replace("{{" + k + "}}", str(v))
        except Exception as e:
            logger.debug("Text interpolation error for %s/%s: %s", category, key, e)
    return text.strip()


def _get_cite(registry: Dict[str, Any], category: str, key: str) -> Optional[str]:
    """Get citation key for a registry entry."""
    cat = registry.get(category, {})
    entry = cat.get(key, {})
    return entry.get("cite")


def _format_param_text(
    registry: Dict[str, Any], params: Dict[str, Any]
) -> str:
    """Generate human-readable parameter description from model params.

    Args:
        registry: The loaded descriptions dict.
        params: Model parameter dict from config.

    Returns:
        Sentence(s) describing the parameters.
    """
    param_descs = registry.get("param_descriptions", {})
    parts = []
    for key, value in params.items():
        template = param_descs.get(key)
        if template is None:
            # Skip params marked as ~ (null) or unknown
            continue
        parts.append(template.replace("{{value}}", str(value)))

    if not parts:
        return ""
    # Join into a sentence
    if len(parts) == 1:
        return "Specifically, " + parts[0] + "."
    return "Specifically, " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."


def _section_experiment(
    registry: Dict[str, Any], config: ClaryonConfig
) -> str:
    """Generate the experiment framing paragraph."""
    ctx = {
        "name": config.experiment.name,
        "seed": str(config.experiment.seed),
        "results_dir": config.experiment.results_dir,
    }
    return _get_text(registry, "experiment", "default", ctx)


def _section_data(
    registry: Dict[str, Any],
    config: ClaryonConfig,
    n_samples: int = 0,
    n_features: int = 0,
) -> str:
    """Generate the data description paragraph(s)."""
    parts = []

    if config.data.tabular is not None:
        tc = config.data.tabular
        id_note = (
            f"Sample identifiers were taken from the ``{tc.id_col}'' column."
            if tc.id_col else "Sequential sample identifiers were generated."
        )
        ctx = {
            "path": tc.path,
            "n_samples": str(n_samples),
            "n_features": str(n_features),
            "label_col": tc.label_col,
            "id_note": id_note,
        }
        parts.append(_get_text(registry, "data", "tabular", ctx))

    if config.data.imaging is not None:
        ic = config.data.imaging
        ctx = {
            "path": ic.path,
            "mask_pattern": ic.mask_pattern or "*mask*",
        }
        fmt = ic.format if ic.format else "nifti"
        parts.append(_get_text(registry, "data", fmt, ctx))

    # Fusion
    if config.data.tabular and config.data.imaging:
        fusion_key = f"fusion_{config.data.fusion}"
        parts.append(_get_text(registry, "data", fusion_key))

    # Radiomics
    if config.data.radiomics and config.data.radiomics.extract:
        ctx = {"config_path": config.data.radiomics.config or "default"}
        parts.append(_get_text(registry, "radiomics", "pyradiomics", ctx))

    return "\n\n".join(p for p in parts if p)


def _section_cv(registry: Dict[str, Any], config: ClaryonConfig) -> str:
    """Generate the cross-validation description."""
    cv = config.cv
    n_seeds = len(cv.seeds)
    seeds_str = ", ".join(str(s) for s in cv.seeds)

    ctx = {
        "n_folds": str(cv.n_folds),
        "n_seeds": str(n_seeds),
        "seeds_str": seeds_str,
        "test_size": str(cv.test_size),
        "test_pct": str(int(cv.test_size * 100)),
        "outer_folds": str(cv.outer_folds),
        "inner_folds": str(cv.inner_folds),
        "group_col": cv.group_col or "group",
    }

    if cv.strategy == "kfold":
        ctx["total_fits"] = str(cv.n_folds * n_seeds)
    elif cv.strategy == "holdout":
        ctx["total_fits"] = str(n_seeds)
    else:
        ctx["total_fits"] = str(cv.n_folds * n_seeds)

    return _get_text(registry, "cv", cv.strategy, ctx)


def _section_encoding(
    registry: Dict[str, Any],
    config: ClaryonConfig,
    n_features: int = 0,
) -> str:
    """Generate quantum encoding description if quantum models are used."""
    has_quantum = any(m.type == "tabular_quantum" for m in config.models)
    if not has_quantum:
        return ""

    pad_len = 1 << int(math.ceil(math.log2(max(n_features, 1))))
    n_qubits = int(math.log2(pad_len))

    ctx = {
        "n_features": str(n_features),
        "pad_len": str(pad_len),
        "n_qubits": str(n_qubits),
    }
    return _get_text(registry, "encoding", "amplitude", ctx)


def _section_models(
    registry: Dict[str, Any], config: ClaryonConfig
) -> str:
    """Generate per-model description paragraphs."""
    parts = []
    cites_used: List[str] = []

    for model_entry in config.models:
        name = model_entry.name
        params = dict(model_entry.params)

        # Build parameter text
        param_text = _format_param_text(registry, params)

        # Build model-specific context
        ctx = {"param_text": param_text}
        # Add quantum-specific params
        if model_entry.type == "tabular_quantum":
            ctx["n_qubits"] = str(params.get("n_qubits", "auto"))
            ctx["epochs"] = str(params.get("epochs", 10))
            ctx["lr"] = str(params.get("lr", 0.02))

        text = _get_text(registry, "models", name, ctx)
        if text:
            cite = _get_cite(registry, "models", name)
            if cite:
                text += f" \\cite{{{cite}}}"
                cites_used.append(cite)
            parts.append(text)
        else:
            # Fallback: model class attribute → generic
            try:
                from ..registry import get as _get_model
                model_cls = _get_model("model", name)
                if hasattr(model_cls, "method_description") and model_cls.method_description:
                    text = model_cls.method_description
                    # Interpolate params
                    if ctx:
                        for k, v in ctx.items():
                            text = text.replace("{{" + k + "}}", str(v))
                    if hasattr(model_cls, "method_cite_key") and model_cls.method_cite_key:
                        text += f" \\cite{{{model_cls.method_cite_key}}}"
                    parts.append(text)
                else:
                    parts.append(
                        f"The \\textbf{{{_latex_escape(name)}}} model was included "
                        f"in the comparison. {param_text}"
                    )
            except (KeyError, ImportError):
                parts.append(
                    f"The \\textbf{{{_latex_escape(name)}}} model was included "
                    f"in the comparison. {param_text}"
                )

    return "\n\n".join(parts)


def _section_metrics(
    registry: Dict[str, Any], config: ClaryonConfig
) -> str:
    """Generate metrics description paragraph."""
    parts = []
    for metric_name in config.evaluation.metrics:
        text = _get_text(registry, "metrics", metric_name)
        cite = _get_cite(registry, "metrics", metric_name)
        if text:
            if cite:
                text += f" \\cite{{{cite}}}"
            parts.append(text)

    # Threshold optimization (always included for binary)
    threshold_text = _get_text(registry, "threshold", "youden")
    if threshold_text:
        parts.append(threshold_text)

    return " ".join(parts)


def _section_statistical_tests(
    registry: Dict[str, Any], config: ClaryonConfig
) -> str:
    """Generate statistical test description."""
    parts = []
    for test_name in config.evaluation.statistical_tests:
        ctx = {
            "confidence_pct": str(int(config.evaluation.confidence_level * 100)),
            "n_bootstrap": "1000",
        }
        text = _get_text(registry, "statistical_tests", test_name, ctx)
        cite = _get_cite(registry, "statistical_tests", test_name)
        if text:
            if cite:
                text += f" \\cite{{{cite}}}"
            parts.append(text)
    return " ".join(parts)


def _section_explainability(
    registry: Dict[str, Any], config: ClaryonConfig
) -> str:
    """Generate explainability description."""
    parts = []
    ctx = {
        "max_test_samples": str(config.explainability.max_test_samples),
        "max_features": str(config.explainability.max_features),
    }

    if config.explainability.shap:
        text = _get_text(registry, "explainability", "shap", ctx)
        cite = _get_cite(registry, "explainability", "shap")
        if text:
            if cite:
                text += f" \\cite{{{cite}}}"
            parts.append(text)

    if config.explainability.lime:
        text = _get_text(registry, "explainability", "lime", ctx)
        cite = _get_cite(registry, "explainability", "lime")
        if text:
            if cite:
                text += f" \\cite{{{cite}}}"
            parts.append(text)

    if config.explainability.grad_cam:
        text = _get_text(registry, "explainability", "gradcam", ctx)
        cite = _get_cite(registry, "explainability", "gradcam")
        if text:
            if cite:
                text += f" \\cite{{{cite}}}"
            parts.append(text)

    return "\n\n".join(parts)


def _collect_citations(registry: Dict[str, Any], config: ClaryonConfig) -> List[str]:
    """Collect all BibTeX cite keys referenced by the experiment config."""
    cites = set()
    for m in config.models:
        c = _get_cite(registry, "models", m.name)
        if c:
            cites.add(c)
    for metric in config.evaluation.metrics:
        c = _get_cite(registry, "metrics", metric)
        if c:
            cites.add(c)
    for test in config.evaluation.statistical_tests:
        c = _get_cite(registry, "statistical_tests", test)
        if c:
            cites.add(c)
    if config.explainability.shap:
        c = _get_cite(registry, "explainability", "shap")
        if c:
            cites.add(c)
    if config.explainability.lime:
        c = _get_cite(registry, "explainability", "lime")
        if c:
            cites.add(c)
    if config.data.radiomics and config.data.radiomics.extract:
        c = _get_cite(registry, "radiomics", "pyradiomics")
        if c:
            cites.add(c)
    has_quantum = any(m.type == "tabular_quantum" for m in config.models)
    if has_quantum:
        c = _get_cite(registry, "encoding", "amplitude")
        if c:
            cites.add(c)
    return sorted(cites)


def generate_bibtex(
    descriptions_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Copy the master references.bib to the output directory.

    The master .bib file contains entries for all cite keys referenced
    in method_descriptions.yaml. This function copies it to the results
    directory so LaTeX can find it.

    Args:
        descriptions_path: Path to method_descriptions.yaml (unused, kept for API compat).
        output_path: Output .bib file path.

    Returns:
        Path to the written .bib file.
    """
    import shutil
    master_bib = Path(__file__).parent / "references.bib"
    if output_path is None:
        output_path = Path("references.bib")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(master_bib, output_path)
    logger.info("Copied references.bib to %s", output_path)
    return output_path


def generate_structured_methods(
    config: ClaryonConfig,
    output_path: Path,
    n_samples: int = 0,
    n_features: int = 0,
) -> Path:
    """Generate a structured LaTeX methods section from the experiment config.

    Args:
        config: Validated experiment configuration.
        output_path: Output .tex file path.
        n_samples: Number of samples in the dataset (for description).
        n_features: Number of features (for encoding description).

    Returns:
        Path to the written .tex file.
    """
    registry = _load_descriptions()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = []

    # 1. Section header
    sections.append(r"\section{Methods}")
    sections.append("")

    # 2. Experiment framing
    sections.append(r"\subsection{Experimental Setup}")
    sections.append(_section_experiment(registry, config))
    sections.append("")

    # 3. Data
    data_text = _section_data(registry, config, n_samples, n_features)
    if data_text:
        sections.append(r"\subsection{Data}")
        sections.append(data_text)
        sections.append("")

    # 4. Preprocessing + encoding
    preproc_parts = []
    prep_cfg = config.preprocessing
    if prep_cfg.zscore:
        zscore_text = _get_text(registry, "preprocessing", "scaling")
        if zscore_text:
            preproc_parts.append(zscore_text)
    if prep_cfg.feature_selection:
        mrmr_text = (
            f"Redundant features (Spearman $|\\rho| > {prep_cfg.spearman_threshold}$) "
            f"were clustered, and within each cluster the feature with the highest "
            f"relevance to the target label was retained (mRMR feature selection). "
            f"Feature selection was performed on the training fold only to prevent "
            f"information leakage."
        )
        preproc_parts.append(mrmr_text)
    if config.binary_grouping is not None and config.binary_grouping.enabled:
        bg = config.binary_grouping
        preproc_parts.append(
            f"Labels were binarized prior to cross-validation: original classes "
            f"{bg.positive} were mapped to positive (1) and "
            f"{bg.negative if bg.negative else 'all remaining classes'} to negative (0)."
        )
    # HF-031: quantum models skip z-score
    has_quantum = any(m.type == "tabular_quantum" for m in config.models)
    if has_quantum:
        quantum_zscore_text = _get_text(registry, "preprocessing", "quantum_no_zscore")
        if quantum_zscore_text:
            preproc_parts.append(quantum_zscore_text)
    encoding_text = _section_encoding(registry, config, n_features)
    if encoding_text:
        preproc_parts.append(encoding_text)
    if preproc_parts:
        sections.append(r"\subsection{Preprocessing and Encoding}")
        sections.append("\n\n".join(preproc_parts))
        sections.append("")

    # 5. Cross-validation
    cv_text = _section_cv(registry, config)
    if cv_text:
        sections.append(r"\subsection{Cross-Validation}")
        sections.append(cv_text)
        sections.append("")

    # 6. Models
    sections.append(r"\subsection{Models}")
    n_models = len(config.models)
    model_names = [m.name for m in config.models]
    model_names_str = ", ".join(model_names[:-1]) + f", and {model_names[-1]}" if n_models > 1 else model_names[0]
    sections.append(
        f"A total of {n_models} models were compared: {model_names_str}."
    )
    sections.append("")
    sections.append(_section_models(registry, config))
    sections.append("")

    # 6b. Ensemble (if more than 1 model)
    if len(config.models) > 1:
        ensemble_text = _get_text(registry, "models", "ensemble")
        if ensemble_text:
            model_list = ", ".join(m.name for m in config.models)
            sections.append(
                f"An ensemble prediction was computed by softmax averaging of the "
                f"per-class probability vectors from {model_list}."
            )
            sections.append("")

    # 7. Evaluation metrics
    sections.append(r"\subsection{Evaluation Metrics}")
    sections.append(_section_metrics(registry, config))
    sections.append("")

    # 8. Statistical tests
    stat_text = _section_statistical_tests(registry, config)
    if stat_text:
        sections.append(r"\subsection{Statistical Analysis}")
        sections.append(stat_text)
        sections.append("")

    # 9. Explainability
    explain_text = _section_explainability(registry, config)
    if explain_text:
        sections.append(r"\subsection{Explainability}")
        sections.append(explain_text)
        sections.append("")

    # 10. Software
    sections.append(r"\subsection{Software}")
    sections.append(
        r"All experiments were conducted using the CLARYON framework "
        r"(v0.8.0) \cite{claryon2026}, running Python~3.11 on an "
        r"Ubuntu Linux server. Quantum circuits were simulated using "
        r"PennyLane's \texttt{default.qubit} backend."
    )
    sections.append("")

    # Assemble
    full_text = "\n".join(sections)

    output_path.write_text(full_text)
    logger.info("Wrote structured methods to %s", output_path)

    # Also write a .bib stub with cited keys
    cites = _collect_citations(registry, config)
    if cites:
        bib_path = output_path.parent / "references_needed.txt"
        bib_path.write_text(
            "% BibTeX entries needed for this methods section:\n"
            + "\n".join(f"% {c}" for c in cites)
            + "\n"
        )
        logger.info("Citation keys written to %s", bib_path)

    return output_path
