"""Preset resolution for model hyperparameters."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_PRESETS_PATH = Path(__file__).parent / "presets.yaml"
_PRESET_LEVELS = ("quick", "small", "medium", "large", "exhaustive")
_QUANTUM_WARN_LEVELS = ("quick", "small")

_cache: Optional[Dict[str, Any]] = None


def _load_presets() -> Dict[str, Any]:
    """Load and cache the presets YAML."""
    global _cache
    if _cache is None:
        with open(_PRESETS_PATH) as f:
            _cache = yaml.safe_load(f)
    return _cache


def resolve_preset(
    model_name: str,
    model_type: str,
    preset_level: str,
) -> Dict[str, Any]:
    """Resolve preset parameters for a model at a given complexity level.

    Merges category defaults with per-model overrides.

    Args:
        model_name: Model registry name (e.g. "xgboost").
        model_type: Model category ("tabular", "tabular_quantum", "imaging").
        preset_level: One of quick/small/medium/large/exhaustive.

    Returns:
        Dict of hyperparameters from the preset.
    """
    presets = _load_presets()

    # Start with category defaults
    defaults = presets.get("_defaults", {})
    category_presets = defaults.get(model_type, {})
    params = dict(category_presets.get(preset_level, {}))

    # Merge per-model overrides
    model_overrides = presets.get(model_name, {})
    level_overrides = model_overrides.get(preset_level, {})
    params.update(level_overrides)

    # Remove null values (e.g. shots: null)
    params = {k: v for k, v in params.items() if v is not None}

    return params


def resolve_model_params(
    model_name: str,
    model_type: str,
    explicit_params: Dict[str, Any],
    model_preset: Optional[str],
    global_complexity: str,
) -> Dict[str, Any]:
    """Full resolution: explicit params > model preset > global complexity > medium default.

    Args:
        model_name: Model registry name.
        model_type: Model category.
        explicit_params: User-specified params from config YAML.
        model_preset: Per-model preset level (or None).
        global_complexity: Global complexity from experiment config.

    Returns:
        Fully resolved hyperparameters.
    """
    # Determine effective preset level
    if model_preset is not None:
        level = model_preset
    elif global_complexity in _PRESET_LEVELS:
        level = global_complexity
    else:
        level = "medium"

    # Get preset params
    preset_params = resolve_preset(model_name, model_type, level)

    # Explicit params override preset
    preset_params.update(explicit_params)

    # Warn for quantum models at low presets
    if model_type == "tabular_quantum" and level in _QUANTUM_WARN_LEVELS:
        epochs = preset_params.get("epochs", 0)
        logger.warning(
            "WARNING: %s using '%s' preset (%d epochs). "
            "For publishable results, use 'medium' or higher.",
            model_name, level, epochs,
        )

    return preset_params
