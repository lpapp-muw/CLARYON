"""Auto complexity mode — select presets based on dataset size and time budget."""
from __future__ import annotations

import logging
from math import ceil, log2
from typing import Any, Dict, List

from ..config_schema import ClaryonConfig
from .preset_resolver import resolve_preset

logger = logging.getLogger(__name__)

_PRESET_ORDER = ["exhaustive", "large", "medium", "small", "quick"]


def estimate_runtime(
    model_name: str,
    model_type: str,
    n_samples: int,
    n_qubits: int,
    params: Dict[str, Any],
) -> float:
    """Estimate seconds per fold for a model.

    Args:
        model_name: Model registry name.
        model_type: Model category.
        n_samples: Number of training samples.
        n_qubits: Number of qubits (for quantum models).
        params: Resolved hyperparameters.

    Returns:
        Estimated runtime in seconds.
    """
    if model_type == "tabular":
        return 5.0  # classical models are essentially instant

    circuit_cost = (2 ** n_qubits) * 1e-4  # seconds per circuit eval (simulator)

    if model_name in ("kernel_svm", "sq_kernel_svm"):
        return n_samples ** 2 * circuit_cost
    elif model_name == "qdc_hadamard":
        return n_samples ** 2 * circuit_cost
    elif model_name == "qdc_swap":
        swap_cost = (2 ** (2 * n_qubits + 1)) * 1e-4
        return n_samples ** 2 * swap_cost
    elif model_name == "quantum_gp":
        return n_samples ** 2 * circuit_cost
    elif model_type == "imaging":
        epochs = params.get("epochs", 50)
        batch_size = params.get("batch_size", 4)
        batches = max(n_samples // batch_size, 1)
        return epochs * batches * 0.5  # ~0.5s per batch on GPU
    else:
        # Training-based quantum models (qcnn_muw, qcnn_alt, vqc, hybrid, qnn)
        epochs = params.get("epochs", 100)
        batch_size = params.get("batch_size", 16)
        batches = max(n_samples // batch_size, 1)
        return epochs * batches * circuit_cost * 3  # 3x for gradients


def auto_select_presets(
    config: ClaryonConfig,
    n_samples: int,
    n_features: int,
    n_features_after_mrmr: int,
) -> Dict[str, str]:
    """Select preset per model based on dataset size and time budget.

    Args:
        config: Experiment configuration.
        n_samples: Number of samples.
        n_features: Original feature count.
        n_features_after_mrmr: Feature count after selection.

    Returns:
        Dict mapping model_name to preset_name.
    """
    n_qubits = ceil(log2(max(n_features_after_mrmr, 2)))
    budget_seconds = config.experiment.max_runtime_minutes * 60
    n_folds = config.cv.n_folds if config.cv.strategy != "holdout" else 1
    n_seeds = len(config.cv.seeds)
    n_models = len(config.models)

    budget_per_model_fold = budget_seconds / max(n_models * n_folds * n_seeds, 1)

    selected: Dict[str, str] = {}
    for model_entry in config.models:
        estimates: Dict[str, float] = {}
        for preset_name in _PRESET_ORDER:
            params = resolve_preset(model_entry.name, model_entry.type, preset_name)
            est = estimate_runtime(
                model_entry.name, model_entry.type, n_samples, n_qubits, params,
            )
            estimates[preset_name] = est

        # Pick highest quality that fits budget
        chosen = "quick"
        for preset_name in _PRESET_ORDER:
            if estimates[preset_name] <= budget_per_model_fold:
                chosen = preset_name
                break

        # Classical models always get at least medium
        if model_entry.type == "tabular" and chosen in ("quick", "small"):
            chosen = "medium"

        selected[model_entry.name] = chosen
        logger.info(
            "  auto: %s → %s (est. %.1fs/fold, budget %.1fs/fold)",
            model_entry.name, chosen,
            estimates[chosen], budget_per_model_fold,
        )

    return selected
