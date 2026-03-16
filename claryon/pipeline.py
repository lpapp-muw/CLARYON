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
    fitted_models: Dict[str, Any] = field(default_factory=dict)
    metrics_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
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


def _load_nifti_volumes(root: str, mask_pattern: Optional[str]) -> Optional[Dataset]:
    """Load NIfTI volumes as 5D array (N, C, D, H, W) for CNN models.

    Args:
        root: Root directory with NIfTI files.
        mask_pattern: Glob pattern for masks.

    Returns:
        Dataset with 5D volume array, or None if no files found.
    """
    from .io.nifti import _collect_pairs, _read_nifti_array, _parse_label, _case_id

    root_path = Path(root)
    # Try Train/ subdirectory first, then root
    train_dir = root_path / "Train"
    search_dir = train_dir if train_dir.exists() else root_path

    pairs = _collect_pairs(search_dir, "*PET*.nii*", mask_pattern)
    if not pairs:
        return None

    volumes = []
    y_labels = []
    ids = []
    for pet_path, mask_path in pairs:
        img = _read_nifti_array(pet_path)
        if mask_path is not None:
            mask = _read_nifti_array(mask_path)
            img = np.where(mask > 0, img, 0.0)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        volumes.append(img)
        y_labels.append(_parse_label(pet_path))
        ids.append(_case_id(pet_path))

    # Pad to uniform shape (max per dimension)
    shapes = [v.shape for v in volumes]
    max_shape = tuple(max(s[d] for s in shapes) for d in range(len(shapes[0])))
    X = np.zeros((len(volumes), 1, *max_shape), dtype=np.float32)
    for i, v in enumerate(volumes):
        slices = tuple(slice(0, s) for s in v.shape)
        X[i, 0][slices] = v

    # Determine task type and encode labels
    unique = sorted(set(y_labels))
    from .io.base import BinaryLabelMapper, MultiClassLabelMapper, TaskType
    if len(unique) == 2:
        mapper = BinaryLabelMapper.fit(y_labels)
        task_type = TaskType.BINARY
    else:
        mapper = MultiClassLabelMapper.fit(y_labels)
        task_type = TaskType.MULTICLASS

    y = mapper.transform(y_labels)
    return Dataset(X=X, y=y, keys=ids, task_type=task_type, label_mapper=mapper)


def stage_load_data(config: ClaryonConfig, state: PipelineState) -> None:
    """Stage 1: Load data from configured sources.

    Args:
        config: Experiment configuration.
        state: Pipeline state to populate with loaded dataset.
    """
    logger.info("Stage 1: Load data")

    ds_tabular: Optional[Dataset] = None
    ds_imaging: Optional[Dataset] = None

    if config.data.tabular is not None:
        from .io.tabular import load_tabular_csv

        tc = config.data.tabular
        ds_tabular = load_tabular_csv(
            tc.path, label_col=tc.label_col, id_col=tc.id_col, sep=tc.sep,
        )
        logger.info("Loaded tabular: %d samples × %d features", ds_tabular.n_samples, ds_tabular.n_features)

    if config.data.imaging is not None:
        ic = config.data.imaging
        mask_pat = ic.mask_pattern if ic.mask_pattern else None

        # Check if any model is imaging type (needs raw 3D volumes)
        has_imaging_model = any(m.type == "imaging" for m in config.models)

        if has_imaging_model:
            # Load as raw 3D volumes (N, C, D, H, W) for CNN models
            ds_imaging = _load_nifti_volumes(ic.path, mask_pat)
        else:
            # Load flattened for tabular-style models
            from .io.nifti import load_nifti_dataset
            nifti_result = load_nifti_dataset(root=ic.path, mask_pattern=mask_pat)
            ds_imaging = nifti_result.get("all") or nifti_result.get("train")

        if ds_imaging is not None:
            logger.info("Loaded imaging: %d samples, shape %s", ds_imaging.n_samples, ds_imaging.X.shape)

    # Combine data sources
    if ds_tabular is not None and ds_imaging is not None:
        # Early fusion: concatenate features
        if ds_tabular.n_samples == ds_imaging.n_samples:
            fused_X = np.concatenate([ds_tabular.X, ds_imaging.X], axis=1)
            fused_names = (ds_tabular.feature_names or [f"tab_{i}" for i in range(ds_tabular.n_features)]) + \
                          [f"img_{i}" for i in range(ds_imaging.n_features)]
            state.dataset = Dataset(
                X=fused_X, y=ds_tabular.y, keys=ds_tabular.keys,
                feature_names=fused_names, task_type=ds_tabular.task_type,
                label_mapper=ds_tabular.label_mapper,
            )
            logger.info("Early fusion: %d samples × %d features", state.dataset.n_samples, state.dataset.n_features)
        else:
            logger.warning("Sample count mismatch (tabular=%d, imaging=%d) — using tabular only",
                           ds_tabular.n_samples, ds_imaging.n_samples)
            state.dataset = ds_tabular
    elif ds_tabular is not None:
        state.dataset = ds_tabular
    elif ds_imaging is not None:
        state.dataset = ds_imaging
    else:
        logger.warning("No data source configured")


def stage_preprocess(config: ClaryonConfig, state: PipelineState) -> None:
    """Stage 2: Preprocess data.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
    """
    logger.info("Stage 2: Preprocess")
    ds = state.dataset
    if ds is None:
        logger.warning("No dataset — skipping preprocessing")
        return

    # Tabular preprocessing: imputation + optional quantile normalization
    if config.data.tabular is not None:
        import pandas as pd
        from .preprocessing.tabular_prep import preprocess_tabular

        feature_names = ds.feature_names or [f"f{i}" for i in range(ds.n_features)]
        df = pd.DataFrame(ds.X, columns=feature_names)
        prep_result = preprocess_tabular(df, use_quantile=False)
        ds.X = prep_result.X
        ds.feature_names = prep_result.feature_names
        logger.info("Tabular preprocessing: %d → %d features", len(feature_names), len(prep_result.feature_names))

    # Radiomics extraction (if configured)
    if config.data.radiomics is not None and config.data.radiomics.extract:
        if config.data.imaging is not None:
            from .io.nifti import _collect_pairs
            from .preprocessing.radiomics import extract_radiomics_batch, merge_radiomics_with_tabular

            ic = config.data.imaging
            pairs = _collect_pairs(
                Path(ic.path),
                pet_pattern="*PET*.nii*",
                mask_pattern=ic.mask_pattern,
            )
            img_mask_pairs = [(str(p), str(m)) for p, m in pairs if m is not None]
            if img_mask_pairs:
                radiomics_config = config.data.radiomics.config
                radiomics_df = extract_radiomics_batch(img_mask_pairs, config_path=radiomics_config)
                if not radiomics_df.empty and ds.keys is not None:
                    feature_names = ds.feature_names or [f"f{i}" for i in range(ds.n_features)]
                    ds.X, ds.feature_names = merge_radiomics_with_tabular(
                        ds.X, feature_names, radiomics_df, ds.keys,
                    )
                    logger.info("Merged radiomics: %d total features", len(ds.feature_names))
            else:
                logger.warning("No image/mask pairs found for radiomics extraction")


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
                    # Amplitude-encode for quantum models
                    X_tr_use, X_te_use = X_train, X_test
                    params = dict(model_entry.params)
                    if model_entry.type == "tabular_quantum":
                        from .encoding.amplitude import amplitude_encode_matrix
                        X_tr_use, enc_info = amplitude_encode_matrix(X_train)
                        X_te_use, _ = amplitude_encode_matrix(X_test, pad_len=enc_info.encoded_dim)
                        # Override n_qubits from encoding (authoritative) — HF-010
                        params["n_qubits"] = enc_info.n_qubits
                        logger.info("  Amplitude encoded: %d features -> %d (n_qubits=%d, overridden from encoding)",
                                    X_train.shape[1], enc_info.encoded_dim, enc_info.n_qubits)

                    model = model_cls(**params)
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

                    # Persist last fitted model + data for explainability
                    state.fitted_models[model_name] = {
                        "model": model,
                        "X_train": X_tr_use,
                        "X_test": X_te_use,
                        "y_test": y_test,
                        "model_entry": model_entry,
                    }

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
    logger.info("Stage 5: Evaluate")

    from .io.predictions import read_predictions
    from .registry import get as registry_get

    # Import metric module to register metrics
    from .evaluation import metrics as _metrics_module  # noqa: F811

    metric_names = config.evaluation.metrics
    if not metric_names:
        logger.info("No metrics configured — skipping evaluation")
        return

    summary_rows: List[Dict[str, Any]] = []

    for model_name, model_results in state.results.items():
        fold_metrics: Dict[str, List[float]] = {m: [] for m in metric_names}

        for result in model_results:
            if result.get("status") != "ok":
                continue
            seed = result["seed"]
            fold = result["fold"]
            pred_path = state.results_dir / model_name / f"seed_{seed}" / f"fold_{fold}" / "Predictions.csv"
            if not pred_path.exists():
                logger.warning("Predictions not found: %s", pred_path)
                continue

            df = read_predictions(pred_path)
            y_true = df["Actual"].to_numpy()
            y_pred = df["Predicted"].to_numpy()
            prob_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
            probs = df[prob_cols].to_numpy() if prob_cols else None

            for mname in metric_names:
                try:
                    metric_fn = registry_get("metric", mname)
                    val = metric_fn(y_true, y_pred, probabilities=probs)
                    fold_metrics[mname].append(val)
                except Exception as e:
                    logger.warning("Metric %s failed for %s fold %d: %s", mname, model_name, fold, e)

        # Aggregate: mean ± std
        row: Dict[str, Any] = {"model": model_name}
        for mname in metric_names:
            vals = fold_metrics[mname]
            if vals:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                row[mname] = mean_val
                row[f"{mname}_std"] = std_val
                logger.info("  %s %s: %.4f ± %.4f", model_name, mname, mean_val, std_val)
            else:
                row[mname] = float("nan")
                row[f"{mname}_std"] = float("nan")
        summary_rows.append(row)
        state.metrics_summary[model_name] = {k: v for k, v in row.items() if k != "model"}

    # Write summary CSV
    if summary_rows:
        import pandas as pd
        summary_df = pd.DataFrame(summary_rows)
        summary_path = state.results_dir / "metrics_summary.csv"
        summary_df.to_csv(summary_path, sep=";", index=False)
        logger.info("Wrote metrics summary to %s", summary_path)


def stage_explain(config: ClaryonConfig, state: PipelineState) -> None:
    """Stage 6: Run explainability methods.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
    """
    logger.info("Stage 6: Explain")

    run_shap = config.explainability.shap
    run_lime = config.explainability.lime

    if not run_shap and not run_lime:
        logger.info("No explainability methods configured — skipping")
        return

    if not state.fitted_models:
        logger.warning("No fitted models available — skipping explainability")
        return

    for model_name, model_info in state.fitted_models.items():
        model = model_info["model"]
        X_train = model_info["X_train"]
        X_test = model_info["X_test"]

        ds = state.dataset
        feature_names = ds.feature_names if ds is not None else None

        explain_dir = state.results_dir / model_name / "explanations"
        explain_dir.mkdir(parents=True, exist_ok=True)

        # Determine predict function
        def _make_predict_proba(m: Any) -> Any:
            def predict_proba_fn(X: np.ndarray) -> np.ndarray:
                return m.predict_proba(X)
            return predict_proba_fn

        predict_fn = _make_predict_proba(model)

        if run_shap:
            try:
                from .explainability.shap_ import SHAPExplainer
                shap_exp = SHAPExplainer(
                    max_features=config.explainability.max_features,
                    max_test_samples=config.explainability.max_test_samples,
                )
                shap_result = shap_exp.explain(predict_fn, X_test, feature_names=feature_names, X_train=X_train)
                # Save SHAP values
                np.save(explain_dir / "shap_values.npy", shap_result["shap_values"])
                logger.info("  SHAP explanations saved for %s", model_name)
            except Exception as e:
                logger.error("  SHAP failed for %s: %s", model_name, e)

        if run_lime:
            try:
                from .explainability.lime_ import LIMEExplainer
                lime_exp = LIMEExplainer(
                    max_features=config.explainability.max_features,
                    max_test_samples=config.explainability.max_test_samples,
                )
                lime_result = lime_exp.explain(predict_fn, X_test, feature_names=feature_names, X_train=X_train)
                # Save LIME explanations as JSON
                import json
                with open(explain_dir / "lime_explanations.json", "w") as f:
                    json.dump(lime_result["explanations"], f, indent=2, default=str)
                logger.info("  LIME explanations saved for %s", model_name)
            except Exception as e:
                logger.error("  LIME failed for %s: %s", model_name, e)


def stage_report(config: ClaryonConfig, state: PipelineState) -> None:
    """Stage 7: Generate reports.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
    """
    logger.info("Stage 7: Report")

    if not state.metrics_summary:
        logger.info("No metrics available — skipping report generation")
        return

    metric_names = config.evaluation.metrics
    model_names = list(state.metrics_summary.keys())
    results_rows = []
    for mname in model_names:
        row: Dict[str, Any] = {"model": mname}
        for metric in metric_names:
            row[metric] = state.metrics_summary[mname].get(metric, float("nan"))
        results_rows.append(row)

    if config.reporting.markdown:
        from .reporting.markdown_report import generate_markdown_report
        md_path = state.results_dir / "report.md"
        generate_markdown_report(
            experiment_name=config.experiment.name,
            seed=config.experiment.seed,
            cv_strategy=config.cv.strategy,
            n_folds=config.cv.n_folds,
            models=model_names,
            metrics=metric_names,
            results=results_rows,
            output_path=md_path,
        )

    if config.reporting.latex:
        from .reporting.latex_report import generate_methods_section, generate_results_section
        generate_methods_section(
            experiment_name=config.experiment.name,
            seed=config.experiment.seed,
            cv_strategy=config.cv.strategy,
            n_folds=config.cv.n_folds,
            models=model_names,
            metrics=metric_names,
            output_path=state.results_dir / "methods.tex",
        )
        generate_results_section(
            metrics=metric_names,
            results=results_rows,
            output_path=state.results_dir / "results.tex",
        )


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
