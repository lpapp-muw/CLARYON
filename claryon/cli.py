"""CLARYON command-line interface."""
from __future__ import annotations

import argparse
import logging
import sys

from . import __version__

logger = logging.getLogger(__name__)


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Add the --config argument to a subcommand parser."""
    parser.add_argument(
        "--config", "-c", required=True, help="Path to experiment YAML config"
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Execute a full experiment from config."""
    logger.info("Running full experiment: %s", args.config)
    from .config_schema import load_config
    config = load_config(args.config)
    from .pipeline import run_pipeline
    run_pipeline(config)


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Run preprocessing stage only."""
    logger.info("Preprocessing: %s", args.config)
    print("preprocess: not yet implemented")


def cmd_train(args: argparse.Namespace) -> None:
    """Run training stage only."""
    logger.info("Training: %s", args.config)
    print("train: not yet implemented")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation stage only."""
    logger.info("Evaluating: %s", args.config)
    print("evaluate: not yet implemented")


def cmd_explain(args: argparse.Namespace) -> None:
    """Run explainability stage only."""
    logger.info("Explaining: %s", args.config)
    print("explain: not yet implemented")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate reports."""
    logger.info("Reporting: %s", args.config)
    print("report: not yet implemented")


def cmd_list_models(args: argparse.Namespace) -> None:
    """List all registered models."""
    from .registry import list_registered
    models = list_registered("model")
    if not models:
        print("No models registered.")
    else:
        for name in sorted(models):
            print(f"  {name}")


def cmd_list_metrics(args: argparse.Namespace) -> None:
    """List all registered metrics."""
    from .registry import list_registered
    metrics = list_registered("metric")
    if not metrics:
        print("No metrics registered.")
    else:
        for name in sorted(metrics):
            print(f"  {name}")


def cmd_validate_config(args: argparse.Namespace) -> None:
    """Validate a config file without running."""
    from .config_schema import load_config
    try:
        config = load_config(args.config)
        print(f"Config valid: {config.experiment.name}")
    except Exception as e:
        print(f"Config invalid: {e}", file=sys.stderr)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="claryon",
        description="CLARYON — CLassical-quantum AI for Reproducible Explainable OpeN-source medicine",
    )
    parser.add_argument("--version", action="version", version=f"claryon {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # run
    p = sub.add_parser("run", help="Run a full experiment from config")
    _add_config_arg(p)
    p.set_defaults(func=cmd_run)

    # preprocess
    p = sub.add_parser("preprocess", help="Run preprocessing only")
    _add_config_arg(p)
    p.set_defaults(func=cmd_preprocess)

    # train
    p = sub.add_parser("train", help="Run training only")
    _add_config_arg(p)
    p.add_argument("--model", help="Train a specific model only")
    p.set_defaults(func=cmd_train)

    # evaluate
    p = sub.add_parser("evaluate", help="Run evaluation only")
    _add_config_arg(p)
    p.set_defaults(func=cmd_evaluate)

    # explain
    p = sub.add_parser("explain", help="Run explainability only")
    _add_config_arg(p)
    p.set_defaults(func=cmd_explain)

    # report
    p = sub.add_parser("report", help="Generate reports")
    _add_config_arg(p)
    p.set_defaults(func=cmd_report)

    # list-models
    p = sub.add_parser("list-models", help="List registered models")
    p.set_defaults(func=cmd_list_models)

    # list-metrics
    p = sub.add_parser("list-metrics", help="List registered metrics")
    p.set_defaults(func=cmd_list_metrics)

    # validate-config
    p = sub.add_parser("validate-config", help="Validate a config file")
    _add_config_arg(p)
    p.set_defaults(func=cmd_validate_config)

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.WARNING
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose >= 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
