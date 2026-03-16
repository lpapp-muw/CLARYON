"""Pydantic configuration schema for CLARYON experiments."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Sub-models
# ═══════════════════════════════════════════════════════════════


class ExperimentConfig(BaseModel):
    """Top-level experiment metadata."""

    name: str = "experiment"
    seed: int = 42
    results_dir: str = "Results"


class TabularDataConfig(BaseModel):
    """Tabular data source configuration."""

    path: str
    label_col: str = "label"
    id_col: str = "Key"
    sep: str = ";"


class ImagingDataConfig(BaseModel):
    """Imaging data source configuration."""

    path: str
    format: Literal["nifti", "tiff", "dicom"] = "nifti"
    mask_pattern: Optional[str] = "*mask*"


class RadiomicsConfig(BaseModel):
    """PyRadiomics extraction configuration."""

    extract: bool = False
    config: Optional[str] = None


class DataConfig(BaseModel):
    """Data source configuration."""

    tabular: Optional[TabularDataConfig] = None
    imaging: Optional[ImagingDataConfig] = None
    radiomics: Optional[RadiomicsConfig] = None
    fusion: Literal["early", "late", "intermediate"] = "early"


class CVConfig(BaseModel):
    """Cross-validation strategy configuration."""

    strategy: Literal["kfold", "holdout", "nested", "external", "group_kfold"] = "kfold"
    n_folds: int = Field(default=5, ge=2)
    seeds: List[int] = Field(default_factory=lambda: [42])
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    outer_folds: int = Field(default=5, ge=2)
    inner_folds: int = Field(default=3, ge=2)
    test_path: Optional[str] = None
    group_col: Optional[str] = None


class ModelEntry(BaseModel):
    """Single model configuration entry."""

    name: str
    type: Literal["tabular", "tabular_quantum", "imaging"] = "tabular"
    params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class ExplainConfig(BaseModel):
    """Explainability configuration."""

    shap: bool = False
    lime: bool = False
    grad_cam: bool = False
    max_features: int = Field(default=32, ge=1)
    max_test_samples: int = Field(default=5, ge=1)


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    metrics: List[str] = Field(
        default_factory=lambda: ["bacc", "auc", "sensitivity", "specificity"]
    )
    statistical_tests: List[str] = Field(default_factory=list)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)


class ReportConfig(BaseModel):
    """Reporting configuration."""

    latex: bool = False
    markdown: bool = True
    figures: bool = True
    figure_dpi: int = Field(default=300, ge=72)


# ═══════════════════════════════════════════════════════════════
# Root config
# ═══════════════════════════════════════════════════════════════


class ClaryonConfig(BaseModel):
    """Root configuration for a CLARYON experiment."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    models: List[ModelEntry] = Field(default_factory=list)
    explainability: ExplainConfig = Field(default_factory=ExplainConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    reporting: ReportConfig = Field(default_factory=ReportConfig)

    @field_validator("models")
    @classmethod
    def at_least_one_model_if_provided(cls, v: List[ModelEntry]) -> List[ModelEntry]:
        """Filter out disabled models."""
        return [m for m in v if m.enabled]


def load_config(path: Union[str, Path]) -> ClaryonConfig:
    """Load and validate a YAML config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated ClaryonConfig instance.

    Raises:
        FileNotFoundError: If path does not exist.
        pydantic.ValidationError: If config is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    config = ClaryonConfig(**raw)
    logger.info("Loaded config from %s: experiment=%s", path, config.experiment.name)
    return config
