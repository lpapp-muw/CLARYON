"""Pipeline stage orchestrator — runs experiments end-to-end."""
from __future__ import annotations

import logging
from pathlib import Path

from .config_schema import ClaryonConfig

logger = logging.getLogger(__name__)


def stage_load_data(config: ClaryonConfig) -> None:
    """Stage 1: Load data from configured sources."""
    logger.info("Stage 1: Load data (stub)")


def stage_preprocess(config: ClaryonConfig) -> None:
    """Stage 2: Preprocess data (imputation, encoding, scaling, radiomics)."""
    logger.info("Stage 2: Preprocess (stub)")


def stage_split(config: ClaryonConfig) -> None:
    """Stage 3: Generate cross-validation splits."""
    logger.info("Stage 3: Split (stub)")


def stage_train(config: ClaryonConfig) -> None:
    """Stage 4: Train all configured models across folds/seeds."""
    logger.info("Stage 4: Train (stub)")


def stage_evaluate(config: ClaryonConfig) -> None:
    """Stage 5: Evaluate models and compute metrics."""
    logger.info("Stage 5: Evaluate (stub)")


def stage_explain(config: ClaryonConfig) -> None:
    """Stage 6: Run explainability methods."""
    logger.info("Stage 6: Explain (stub)")


def stage_report(config: ClaryonConfig) -> None:
    """Stage 7: Generate reports (LaTeX, Markdown, figures)."""
    logger.info("Stage 7: Report (stub)")


def run_pipeline(config: ClaryonConfig) -> None:
    """Execute all pipeline stages in order.

    Args:
        config: Validated experiment configuration.
    """
    logger.info("Pipeline start: experiment=%s", config.experiment.name)

    from .determinism import enforce_determinism
    enforce_determinism(config.experiment.seed)

    stages = [
        ("load_data", stage_load_data),
        ("preprocess", stage_preprocess),
        ("split", stage_split),
        ("train", stage_train),
        ("evaluate", stage_evaluate),
        ("explain", stage_explain),
        ("report", stage_report),
    ]

    for name, fn in stages:
        logger.info("=== Stage: %s ===", name)
        fn(config)

    logger.info("Pipeline complete: experiment=%s", config.experiment.name)
