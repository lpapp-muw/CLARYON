"""Pipeline stage orchestrator — runs experiments end-to-end."""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config_schema import ClaryonConfig
from .io.base import BinaryLabelMapper, Dataset, TaskType
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
        "claryon.models.quantum.sq_kernel_svm",
        "claryon.models.quantum.qdc_hadamard",
        "claryon.models.quantum.qdc_swap",
        "claryon.models.quantum.quantum_gp",
        "claryon.models.quantum.qnn",
    ]
    for mod in modules:
        importlib.import_module(mod)
    for mod in optional:
        try:
            importlib.import_module(mod)
        except ImportError:
            logger.debug("Optional model module not available: %s", mod)


def _load_nifti_volumes(
    root: str,
    mask_pattern: Optional[str],
    image_pattern: str = "*.nii*",
) -> Optional[Dataset]:
    """Load NIfTI volumes as 5D array (N, C, D, H, W) for CNN models.

    Args:
        root: Root directory with NIfTI files.
        mask_pattern: Glob pattern for masks.
        image_pattern: Glob pattern for image volumes.

    Returns:
        Dataset with 5D volume array, or None if no files found.
    """
    from .io.nifti import _collect_pairs, _read_nifti_array, _parse_label, _case_id

    root_path = Path(root)
    # Try Train/ subdirectory first, then root
    train_dir = root_path / "Train"
    search_dir = train_dir if train_dir.exists() else root_path

    pairs = _collect_pairs(search_dir, image_pattern, mask_pattern)
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
    from .io.base import MultiClassLabelMapper
    if len(unique) == 2:
        mapper = BinaryLabelMapper.fit(y_labels)
        task_type = TaskType.BINARY
    else:
        mapper = MultiClassLabelMapper.fit(y_labels)
        task_type = TaskType.MULTICLASS

    y = mapper.transform(y_labels)
    return Dataset(X=X, y=y, keys=ids, task_type=task_type, label_mapper=mapper)


def stage_load_data(config: ClaryonConfig, state: PipelineState, **kwargs: Any) -> None:
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

        img_pat = ic.image_pattern if ic.image_pattern else "*.nii*"

        if has_imaging_model:
            # Load as raw 3D volumes (N, C, D, H, W) for CNN models
            ds_imaging = _load_nifti_volumes(ic.path, mask_pat, image_pattern=img_pat)
        else:
            # Load flattened for tabular-style models
            from .io.nifti import load_nifti_dataset
            nifti_result = load_nifti_dataset(root=ic.path, pet_pattern=img_pat, mask_pattern=mask_pat)
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


def stage_binary_grouping(config: ClaryonConfig, state: PipelineState, **kwargs: Any) -> None:
    """Stage 2: Apply binary grouping if configured.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
    """
    logger.info("Stage 2: Binary grouping")
    ds = state.dataset
    if ds is None or ds.y is None:
        return

    if config.binary_grouping is not None and config.binary_grouping.enabled:
        from .preprocessing.binary_grouping import apply_binary_grouping

        ds.y = apply_binary_grouping(ds.y, config.binary_grouping)
        ds.task_type = TaskType.BINARY
        ds.label_mapper = BinaryLabelMapper(
            classes=[0, 1],
            to_int={0: 0, 1: 1},
            to_label={0: 0, 1: 1},
        )
        logger.info("Binary grouping applied: %d pos / %d neg",
                     int(ds.y.sum()), int((ds.y == 0).sum()))
    else:
        logger.info("Binary grouping not configured — skipping")


def stage_preprocess(config: ClaryonConfig, state: PipelineState, **kwargs: Any) -> None:
    """Stage 3: Preprocess data (imputation + radiomics, NOT z-score/mRMR).

    Z-score and mRMR happen per-fold inside stage_train to prevent data leakage.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
    """
    logger.info("Stage 3: Preprocess (imputation/radiomics)")
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
                pet_pattern=ic.image_pattern or "*.nii*",
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


def stage_split(config: ClaryonConfig, state: PipelineState, **kwargs: Any) -> None:
    """Stage 4: Generate cross-validation splits.

    Args:
        config: Experiment configuration.
        state: Pipeline state to populate with splits.
    """
    logger.info("Stage 4: Split")
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


def _preprocess_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    config: ClaryonConfig,
    model_type: str = "tabular",
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Apply per-fold preprocessing: mRMR + optional z-score on training data.

    Z-score is applied only to classical (tabular) models. Quantum models
    skip z-score because amplitude encoding handles normalization, and
    z-score distorts quantum kernel geometry (HF-031).

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        feature_names: Feature names.
        config: Experiment configuration.
        model_type: Model type ("tabular", "tabular_quantum", "imaging").

    Returns:
        (X_train_preprocessed, X_test_preprocessed, PreprocessingState)
    """
    from .preprocessing.feature_selection import mrmr_select
    from .preprocessing.state import PreprocessingState
    from .preprocessing.tabular_prep import apply_zscore, fit_zscore

    prep_cfg = config.preprocessing

    # Step 1: Always compute z-score coefficients (needed for classical inference)
    if prep_cfg.zscore:
        z_mean, z_std = fit_zscore(X_train)
    else:
        z_mean = np.zeros(X_train.shape[1])
        z_std = np.ones(X_train.shape[1])

    # Step 2: mRMR feature selection (applies to ALL tabular models)
    if prep_cfg.feature_selection:
        selected_idx, selected_names = mrmr_select(
            X_train, y_train, feature_names,
            spearman_threshold=prep_cfg.spearman_threshold,
            max_features=prep_cfg.max_features,
        )
    else:
        selected_idx = list(range(X_train.shape[1]))
        selected_names = feature_names[:X_train.shape[1]]

    # Select features first (before z-score, so quantum gets raw selected)
    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    # Step 3: Apply z-score only to classical models (HF-031)
    if model_type == "tabular_quantum":
        # Quantum: mRMR only, NO z-score
        # Amplitude encoding handles normalization
        X_train_out = X_train_sel
        X_test_out = X_test_sel
        preprocessing_applied = "mrmr_only"
        logger.info("  Quantum model: skipping z-score (amplitude encoding normalizes)")
    else:
        # Classical: z-score + mRMR
        if prep_cfg.zscore:
            z_mean_sel = z_mean[selected_idx]
            z_std_sel = z_std[selected_idx]
            X_train_out = apply_zscore(X_train_sel, z_mean_sel, z_std_sel)
            X_test_out = apply_zscore(X_test_sel, z_mean_sel, z_std_sel)
        else:
            X_train_out = X_train_sel
            X_test_out = X_test_sel
        preprocessing_applied = "zscore_mrmr" if prep_cfg.zscore else "mrmr_only"

    preproc_state = PreprocessingState(
        z_mean=z_mean,
        z_std=z_std,
        selected_features=selected_idx,
        selected_feature_names=selected_names,
        spearman_threshold=prep_cfg.spearman_threshold,
        image_norm_mode=prep_cfg.image_normalization,
        n_features_original=len(feature_names),
        n_features_selected=len(selected_idx),
        preprocessing_applied=preprocessing_applied,
    )

    return X_train_out, X_test_out, preproc_state


def stage_train(config: ClaryonConfig, state: PipelineState, **kwargs: Any) -> None:
    """Stage 5: Train all configured models across folds/seeds.

    Preprocessing (z-score, mRMR) happens INSIDE each fold to prevent leakage.

    Args:
        config: Experiment configuration.
        state: Pipeline state with dataset and splits.
    """
    logger.info("Stage 5: Train (with per-fold preprocessing)")

    progress = kwargs.get("progress")
    _import_model_modules()

    ds = state.dataset
    if ds is None or ds.y is None:
        logger.warning("No dataset — skipping training")
        return

    state.results_dir = Path(config.experiment.results_dir)
    state.results_dir.mkdir(parents=True, exist_ok=True)

    feature_names = ds.feature_names or [f"f{i}" for i in range(ds.n_features)]
    is_tabular = ds.X.ndim == 2

    # Auto complexity resolution (deferred until first fold provides n_features_after_mrmr)
    auto_presets: Optional[Dict[str, str]] = None

    for model_entry in config.models:
        model_name = model_entry.name
        logger.info("Training model: %s", model_name)

        # Model/data type validation
        if model_entry.type == "imaging" and is_tabular:
            logger.error(
                "SKIPPING %s: model type 'imaging' requires imaging data "
                "(config.data.imaging). Cannot run CNN on tabular features.",
                model_name,
            )
            continue
        if model_entry.type in ("tabular", "tabular_quantum") and not is_tabular:
            logger.error(
                "SKIPPING %s: model type '%s' requires tabular data "
                "(config.data.tabular). Cannot run on imaging data alone.",
                model_name, model_entry.type,
            )
            continue

        try:
            model_cls = get("model", model_name)
        except KeyError:
            logger.error("Model %r not registered — skipping", model_name)
            continue

        model_results: List[Dict[str, Any]] = []
        _t_model_start = time.monotonic()
        total_folds = sum(len(s) for s in state.splits.values())
        fold_counter = 0

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
                    pred_dir = state.results_dir / model_name / f"seed_{seed}" / f"fold_{fold}"

                    # Per-fold preprocessing for tabular data
                    preproc_state = None
                    if is_tabular and model_entry.type != "imaging":
                        X_train, X_test, preproc_state = _preprocess_fold(
                            X_train, X_test, y_train, feature_names, config,
                            model_type=model_entry.type,
                        )
                        logger.info("  Preprocessed: %d → %d features",
                                    preproc_state.n_features_original,
                                    preproc_state.n_features_selected)
                    elif model_entry.type == "imaging":
                        # Image normalization
                        from .preprocessing.image_prep import normalize_images
                        mode = config.preprocessing.image_normalization
                        if mode == "cohort_global":
                            gmin = float(X_train.min())
                            gmax = float(X_train.max())
                            X_train, _, _ = normalize_images(X_train, mode="cohort_global",
                                                             global_min=gmin, global_max=gmax)
                            X_test, _, _ = normalize_images(X_test, mode="cohort_global",
                                                            global_min=gmin, global_max=gmax)
                        else:
                            X_train, _, _ = normalize_images(X_train, mode="per_image")
                            X_test, _, _ = normalize_images(X_test, mode="per_image")

                    # Auto complexity: resolve once on first fold
                    if config.experiment.complexity == "auto" and auto_presets is None:
                        from .models.auto_complexity import auto_select_presets
                        n_feat_after = (preproc_state.n_features_selected
                                        if preproc_state else X_train.shape[1])
                        auto_presets = auto_select_presets(
                            config, ds.n_samples, ds.n_features, n_feat_after,
                        )

                    # Amplitude-encode for quantum models
                    X_tr_use, X_te_use = X_train, X_test

                    # Resolve preset parameters
                    from .models.preset_resolver import resolve_model_params
                    effective_complexity = config.experiment.complexity
                    if auto_presets and model_entry.name in auto_presets:
                        effective_complexity = auto_presets[model_entry.name]
                    params = resolve_model_params(
                        model_name=model_entry.name,
                        model_type=model_entry.type,
                        explicit_params=dict(model_entry.params),
                        model_preset=model_entry.preset,
                        global_complexity=effective_complexity,
                    )
                    if model_entry.type == "tabular_quantum":
                        from .encoding.amplitude import amplitude_encode_matrix
                        X_tr_use, enc_info = amplitude_encode_matrix(X_train)
                        X_te_use, _ = amplitude_encode_matrix(X_test, pad_len=enc_info.encoded_dim)
                        # Override n_qubits from encoding (authoritative) — HF-010
                        params["n_qubits"] = enc_info.n_qubits
                        logger.info("  Amplitude encoded: %d features -> %d (n_qubits=%d)",
                                    X_train.shape[1], enc_info.encoded_dim, enc_info.n_qubits)

                    # Preflight resource check
                    n_qubits = params.get("n_qubits", 0)
                    if model_entry.type in ("tabular_quantum",):
                        from .safety import preflight_resource_check, get_available_memory_gb, estimate_memory_gb
                        warnings = preflight_resource_check(
                            model_name, model_entry.type,
                            len(y_train), n_qubits, params,
                        )
                        for w in warnings:
                            logger.warning("  %s", w)
                        # Check if estimated memory exceeds 80% of available
                        est_gb = estimate_memory_gb(model_name, n_qubits, len(y_train))
                        avail_gb = get_available_memory_gb()
                        if est_gb > 0.8 * avail_gb:
                            logger.error(
                                "SKIPPING %s: estimated memory %.1f GB exceeds "
                                "80%% of available %.1f GB.",
                                model_name, est_gb, avail_gb,
                            )
                            model_results.append({
                                "seed": seed, "fold": fold,
                                "status": "skipped_memory",
                            })
                            continue

                    try:
                        model = model_cls(**params)
                    except TypeError:
                        # Fallback: use only explicit user params if preset params conflict
                        model = model_cls(**dict(model_entry.params))
                    model.fit(X_tr_use, y_train, ds.task_type)

                    if ds.task_type == TaskType.REGRESSION:
                        predicted = model.predict(X_te_use)
                        probs = None
                    else:
                        probs = model.predict_proba(X_te_use)
                        predicted = np.argmax(probs, axis=1)

                    # Write predictions
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

                    # Save PreprocessingState per fold
                    if preproc_state is not None:
                        preproc_state.save(pred_dir / "preprocessing_state.json")

                    # Save model to disk
                    if hasattr(model, "save"):
                        try:
                            model.save(pred_dir)
                            logger.info("  Model saved to %s", pred_dir)
                        except Exception as save_err:
                            logger.warning("  Model save failed: %s", save_err)

                    # Save resolved params
                    import json as _json
                    params_path = pred_dir / "model_params.json"
                    params_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(params_path, "w") as _pf:
                        _json.dump(params, _pf, indent=2, default=str)

                    # Persist last fitted model + data for explainability
                    state.fitted_models[model_name] = {
                        "model": model,
                        "X_train": X_tr_use,
                        "X_test": X_te_use,
                        "y_test": y_test,
                        "model_entry": model_entry,
                        "feature_names": (preproc_state.selected_feature_names
                                          if preproc_state else feature_names),
                    }

                    model_results.append({
                        "seed": seed, "fold": fold,
                        "n_train": len(y_train), "n_test": len(y_test),
                        "status": "ok",
                    })
                    fold_counter += 1
                    if progress is not None:
                        elapsed = time.monotonic() - _t_model_start
                        progress.model_progress(model_name, fold_counter, total_folds, elapsed)

                except MemoryError:
                    logger.error("  OUT OF MEMORY during %s training. Skipping.", model_name)
                    model_results.append({
                        "seed": seed, "fold": fold, "status": "oom",
                    })
                    continue
                except Exception as e:
                    logger.error("  FAILED: %s", e)
                    model_results.append({
                        "seed": seed, "fold": fold, "status": "error", "error": str(e),
                    })

        state.results[model_name] = model_results


def stage_evaluate(config: ClaryonConfig, state: PipelineState, **kwargs: Any) -> None:
    """Stage 6: Evaluate models and compute metrics.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
    """
    logger.info("Stage 6: Evaluate")

    from .io.predictions import read_predictions
    from .registry import get as registry_get

    # Import metric module to register metrics

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
        # Round numeric columns to 6 decimal places for clean output
        numeric_cols = summary_df.select_dtypes(include="number").columns
        summary_df[numeric_cols] = summary_df[numeric_cols].round(6)
        summary_path = state.results_dir / "metrics_summary.csv"
        summary_df.to_csv(summary_path, sep=";", index=False, na_rep="NaN")
        logger.info("Wrote metrics summary to %s", summary_path)


def stage_explain(config: ClaryonConfig, state: PipelineState, **kwargs: Any) -> None:
    """Stage 7: Run explainability methods.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
    """
    logger.info("Stage 7: Explain")

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

        # Use per-fold feature names if available (after mRMR)
        feature_names = model_info.get("feature_names")
        if feature_names is None:
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
                # Generate SHAP plots
                from .explainability.plots import generate_shap_plots
                generate_shap_plots(
                    shap_result["shap_values"],
                    feature_names=feature_names,
                    X_test=X_test,
                    output_dir=explain_dir,
                    dpi=config.reporting.figure_dpi,
                )
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
                # Generate LIME plots
                from .explainability.plots import generate_lime_plots
                generate_lime_plots(
                    lime_result["explanations"],
                    output_dir=explain_dir,
                    dpi=config.reporting.figure_dpi,
                )
                logger.info("  LIME explanations saved for %s", model_name)
            except Exception as e:
                logger.error("  LIME failed for %s: %s", model_name, e)


def stage_report(config: ClaryonConfig, state: PipelineState, **kwargs: Any) -> None:
    """Stage 8: Generate reports.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
    """
    logger.info("Stage 8: Report")

    results_dir = state.results_dir

    # Determine dataset dimensions for method descriptions
    n_samples = state.dataset.n_samples if state.dataset else 0
    n_features = state.dataset.n_features if state.dataset else 0

    # --- Structured methods section (prose from text registry) ---
    if config.reporting.latex:
        try:
            from .reporting.structured_report import generate_structured_methods
            generate_structured_methods(
                config, results_dir / "methods.tex",
                n_samples=n_samples, n_features=n_features,
            )
        except Exception as e:
            logger.warning("Structured methods generation failed: %s — falling back to simple", e)
            from .reporting.latex_report import generate_methods_section
            generate_methods_section(
                experiment_name=config.experiment.name,
                seed=config.experiment.seed,
                cv_strategy=config.cv.strategy,
                n_folds=config.cv.n_folds,
                models=[m.name for m in config.models],
                metrics=config.evaluation.metrics,
                output_path=results_dir / "methods.tex",
            )

    # --- Load metrics from CSV for report generation ---
    metrics_csv = results_dir / "metrics_summary.csv"
    if metrics_csv.exists():
        import pandas as pd
        df = pd.read_csv(metrics_csv, sep=";")
        metric_names = [c for c in df.columns if c != "model" and not c.endswith("_std")]
        results_rows = []
        for _, row in df.iterrows():
            r: Dict[str, Any] = {"model": row["model"]}
            for m in metric_names:
                r[m] = row[m]
                std_col = f"{m}_std"
                if std_col in df.columns:
                    r[std_col] = row[std_col]
            results_rows.append(r)

        # --- Results table (LaTeX) ---
        if config.reporting.latex:
            try:
                from .reporting.latex_report import generate_results_section
                generate_results_section(metric_names, results_rows, results_dir / "results.tex")
            except Exception as e:
                logger.warning("Results table generation failed: %s", e)

        # --- Markdown report ---
        if config.reporting.markdown:
            try:
                from .reporting.markdown_report import generate_markdown_report
                generate_markdown_report(
                    config.experiment.name,
                    config.experiment.seed,
                    config.cv.strategy,
                    config.cv.n_folds,
                    [m.name for m in config.models],
                    metric_names,
                    results_rows,
                    results_dir / "report.md",
                )
            except Exception as e:
                logger.warning("Markdown report generation failed: %s", e)

    if not state.metrics_summary:
        logger.info("No metrics available — skipping report generation")


def _write_provenance(config: ClaryonConfig, state: PipelineState, runtime_seconds: float) -> None:
    """Write provenance metadata to results directory.

    Args:
        config: Experiment configuration.
        state: Pipeline state.
        runtime_seconds: Total pipeline runtime.
    """
    import hashlib
    import json
    import platform
    import socket
    import subprocess
    from datetime import datetime, timezone

    from . import __version__

    results_dir = state.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Git commit (if available)
    git_commit = ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass

    # Config hash
    import yaml
    config_yaml = yaml.dump(config.model_dump(), default_flow_style=False)
    config_hash = "sha256:" + hashlib.sha256(config_yaml.encode()).hexdigest()[:12]

    run_info = {
        "claryon_version": __version__,
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "config_hash": config_hash,
        "git_commit": git_commit,
        "runtime_seconds": round(runtime_seconds, 1),
        "n_models": len(config.models),
        "n_folds": config.cv.n_folds,
        "n_seeds": len(config.cv.seeds),
    }

    with open(results_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    logger.info("Wrote run_info.json to %s", results_dir)

    # Copy config used
    with open(results_dir / "config_used.yaml", "w") as f:
        f.write(config_yaml)


def run_pipeline(config: ClaryonConfig, verbosity: int = 0) -> PipelineState:
    """Execute all pipeline stages in order.

    Args:
        config: Validated experiment configuration.
        verbosity: CLI verbosity level (0=summary, 1=stages, 2=details).

    Returns:
        PipelineState with results from all stages.
    """
    from .progress import ProgressDisplay

    logger.info("Pipeline start: experiment=%s", config.experiment.name)
    t_start = time.monotonic()

    from .determinism import enforce_determinism
    enforce_determinism(config.experiment.seed)

    state = PipelineState()
    progress = ProgressDisplay(verbosity=verbosity, n_stages=8)

    stage_names = [
        "Loading data",
        "Binary grouping",
        "Preprocessing",
        "Splitting",
        "Training",
        "Evaluating",
        "Explaining",
        "Reporting",
    ]
    stage_fns = [
        stage_load_data,
        stage_binary_grouping,
        stage_preprocess,
        stage_split,
        stage_train,
        stage_evaluate,
        stage_explain,
        stage_report,
    ]

    for stage_label, fn in zip(stage_names, stage_fns):
        progress.stage_start(stage_label)
        fn(config, state, progress=progress)
        # Build a short summary for the stage
        detail = ""
        if fn is stage_load_data and state.dataset is not None:
            detail = f"{state.dataset.n_samples} samples × {state.dataset.n_features} features"
        elif fn is stage_split and state.splits:
            total = sum(len(v) for v in state.splits.values())
            detail = f"{total} folds"
        elif fn is stage_train:
            ok = sum(
                1 for res in state.results.values()
                for r in res if r.get("status") == "ok"
            )
            detail = f"{ok} fits completed"
        elif fn is stage_evaluate and state.metrics_summary:
            detail = f"{len(state.metrics_summary)} models scored"
        progress.stage_done(detail)

    runtime = time.monotonic() - t_start
    _write_provenance(config, state, runtime)

    # Print summary table
    metric_names = config.evaluation.metrics
    progress.summary_table(state.metrics_summary, metric_names, str(state.results_dir))

    logger.info("Pipeline complete: experiment=%s (%.1fs)", config.experiment.name, runtime)
    return state
