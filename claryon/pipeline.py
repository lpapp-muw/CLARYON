"""Pipeline stage orchestrator — runs experiments end-to-end."""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config_schema import ClaryonConfig
from .io.base import Dataset, TaskType
from .io.predictions import write_predictions
from .preprocessing.splits import SplitIndices, auto_split
from .registry import get

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Mutable state passed between pipeline stages.

    Attributes:
        dataset: Loaded dataset.
        splits: CV split indices per seed.
        results: Model predictions per model/seed/fold.
        results_dir: Output directory root.
    """

    dataset: Optional[Dataset] = None
    splits: Dict[int, List[SplitIndices]] = field(default_factory=dict)
    results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    results_dir: Path = Path("Results")


def _import_model_modules() -> None:
    """Import model modules to trigger @register decorators."""
    modules = [
        "claryon.models.classical.mlp_",
    ]
    optional = [
        "claryon.models.classical.xgboost_",
        "claryon.models.classical.lightgbm_",
        "claryon.models.classical.catboost_",
        "claryon.models.classical.tabpfn_",
        "claryon.models.classical.debinn_",
        "claryon.models.classical.tabm_",
        "claryon.models.classical.realmlp_",
        "claryon.models.classical.modernnca_",
        "claryon.models.classical.cnn_2d",
        "claryon.models.classical.cnn_3d",
        "claryon.models.quantum.kernel_svm",
        "claryon.models.quantum.qcnn_muw",
        "claryon.models.quantum.qcnn_alt",
        "claryon.models.quantum.vqc",
        "claryon.models.quantum.hybrid",
    ]
    for mod in modules:
        importlib.import_module(mod)
    for mod in optional:
        try:
            importlib.import_module(mod)
        except ImportError:
            logger.debug("Optional model module not available: %s", mod)


def stage_load_data(config: ClaryonConfig, state: PipelineState) -> None:
    """Stage 1: Load data from configured sources.

    Args:
        config: Experiment configuration.
        state: Pipeline state to populate with loaded dataset.
    """
    logger.info("Stage 1: Load data")

    if config.data.tabular is not None:
        from .io.tabular import load_tabular_csv

        tc = config.data.tabular
        ds = load_tabular_csv(
            tc.path, label_col=tc.label_col, id_col=tc.id_col, sep=tc.sep,
        )
        state.dataset = ds
        logger.info("Loaded tabular: %d samples × %d features", ds.n_samples, ds.n_features)
    else:
        logger.warning("No data source configured")


def stage_preprocess(config: ClaryonConfig, state: PipelineState) -> None:
    """Stage 2: Preprocess data.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
    """
    logger.info("Stage 2: Preprocess (passthrough for pre-processed data)")
    # Tabular data loaded via load_tabular_csv is already numeric.
    # Full preprocessing (imputation, quantile norm) is done externally
    # or via a preprocessing-specific config entry. For now, passthrough.


def stage_split(config: ClaryonConfig, state: PipelineState) -> None:
    """Stage 3: Generate cross-validation splits.

    Args:
        config: Experiment configuration.
        state: Pipeline state to populate with splits.
    """
    logger.info("Stage 3: Split")
    ds = state.dataset
    if ds is None or ds.y is None:
        logger.warning("No dataset or labels — skipping split")
        return

    cv = config.cv
    for seed in cv.seeds:
        splits = auto_split(
            ds.y,
            strategy=cv.strategy,
            n_folds=cv.n_folds,
            seed=seed,
            test_size=cv.test_size,
        )
        state.splits[seed] = splits
        logger.info("Generated %d splits for seed=%d", len(splits), seed)


def stage_train(config: ClaryonConfig, state: PipelineState) -> None:
    """Stage 4: Train all configured models across folds/seeds.

    Args:
        config: Experiment configuration.
        state: Pipeline state with dataset and splits.
    """
    logger.info("Stage 4: Train")

    _import_model_modules()

    ds = state.dataset
    if ds is None or ds.y is None:
        logger.warning("No dataset — skipping training")
        return

    state.results_dir = Path(config.experiment.results_dir)
    state.results_dir.mkdir(parents=True, exist_ok=True)

    for model_entry in config.models:
        model_name = model_entry.name
        logger.info("Training model: %s", model_name)

        try:
            model_cls = get("model", model_name)
        except KeyError:
            logger.error("Model %r not registered — skipping", model_name)
            continue

        model_results: List[Dict[str, Any]] = []

        for seed, splits in state.splits.items():
            for split in splits:
                fold = split.fold
                X_train = ds.X[split.train_idx]
                y_train = ds.y[split.train_idx]
                X_test = ds.X[split.test_idx]
                y_test = ds.y[split.test_idx]
                keys_test = [ds.keys[i] for i in split.test_idx] if ds.keys else [
                    f"S{i:04d}" for i in split.test_idx
                ]

                logger.info("  seed=%d fold=%d train=%d test=%d", seed, fold, len(y_train), len(y_test))

                try:
                    model = model_cls(**model_entry.params)

                    # Amplitude-encode for quantum models
                    X_tr_use, X_te_use = X_train, X_test
                    if model_entry.type == "tabular_quantum":
                        from .encoding.amplitude import amplitude_encode_matrix
                        X_tr_use, enc_info = amplitude_encode_matrix(X_train)
                        X_te_use, _ = amplitude_encode_matrix(X_test, pad_len=enc_info.encoded_dim)
                        logger.info("  Amplitude encoded: %d features -> %d (qubits=%d)",
                                    X_train.shape[1], enc_info.encoded_dim, enc_info.n_qubits)

                    model.fit(X_tr_use, y_train, ds.task_type)

                    if ds.task_type == TaskType.REGRESSION:
                        predicted = model.predict(X_test)
                        probs = None
                    else:
                        probs = model.predict_proba(X_test)
                        predicted = np.argmax(probs, axis=1)

                    # Write predictions
                    pred_dir = state.results_dir / model_name / f"seed_{seed}" / f"fold_{fold}"
                    write_predictions(
                        pred_dir / "Predictions.csv",
                        keys=keys_test,
                        actual=y_test,
                        predicted=predicted,
                        probabilities=probs,
                        task_type=ds.task_type,
                        fold=fold,
                        seed=seed,
                    )

                    model_results.append({
                        "seed": seed, "fold": fold,
                        "n_train": len(y_train), "n_test": len(y_test),
                        "status": "ok",
                    })
                except Exception as e:
                    logger.error("  FAILED: %s", e)
                    model_results.append({
                        "seed": seed, "fold": fold, "status": "error", "error": str(e),
                    })

        state.results[model_name] = model_results


def stage_evaluate(config: ClaryonConfig, state: PipelineState) -> None:
    """Stage 5: Evaluate models and compute metrics.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
    """
    logger.info("Stage 5: Evaluate (stub — metrics module not yet implemented)")


def stage_explain(config: ClaryonConfig, state: PipelineState) -> None:
    """Stage 6: Run explainability methods.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
    """
    logger.info("Stage 6: Explain (stub)")


def stage_report(config: ClaryonConfig, state: PipelineState) -> None:
    """Stage 7: Generate reports.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
    """
    logger.info("Stage 7: Report (stub)")


def run_pipeline(config: ClaryonConfig) -> PipelineState:
    """Execute all pipeline stages in order.

    Args:
        config: Validated experiment configuration.

    Returns:
        PipelineState with results from all stages.
    """
    logger.info("Pipeline start: experiment=%s", config.experiment.name)

    from .determinism import enforce_determinism
    enforce_determinism(config.experiment.seed)

    state = PipelineState()

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
        fn(config, state)

    logger.info("Pipeline complete: experiment=%s", config.experiment.name)
    return state
