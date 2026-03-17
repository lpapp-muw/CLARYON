"""SHAP and LIME plot generation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def generate_shap_plots(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]],
    X_test: np.ndarray,
    output_dir: Path,
    dpi: int = 300,
    max_waterfall_samples: int = 3,
) -> List[Path]:
    """Generate SHAP visualization plots.

    Args:
        shap_values: SHAP values array, shape (n_samples, n_features) or (n_samples, n_features, n_classes).
        feature_names: Feature names for axis labels.
        X_test: Test data used for SHAP computation.
        output_dir: Directory to save plots.
        dpi: Figure resolution.
        max_waterfall_samples: Number of waterfall plots to generate.

    Returns:
        List of saved plot paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping SHAP plots")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    # Handle multi-class: use class 1 values for binary
    vals = shap_values
    if vals.ndim == 3:
        vals = vals[:, :, 1] if vals.shape[2] == 2 else vals[:, :, 0]

    n_features = vals.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    feature_names = feature_names[:n_features]

    # Bar plot: mean |SHAP value| per feature
    try:
        mean_abs = np.mean(np.abs(vals), axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1][:20]  # top 20

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            range(len(sorted_idx)),
            mean_abs[sorted_idx][::-1],
            color="#1f77b4",
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP Feature Importance")
        plt.tight_layout()
        bar_path = output_dir / "shap_bar.png"
        fig.savefig(bar_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(bar_path)
        logger.info("  Saved %s", bar_path)
    except Exception as e:
        logger.warning("  SHAP bar plot failed: %s", e)

    # Beeswarm plot: SHAP value distribution per feature
    try:
        sorted_idx = np.argsort(np.mean(np.abs(vals), axis=0))[::-1][:15]

        fig, ax = plt.subplots(figsize=(10, 8))
        for rank, feat_idx in enumerate(sorted_idx[::-1]):
            y_positions = np.full(vals.shape[0], rank)
            y_positions += np.random.default_rng(42).uniform(-0.3, 0.3, size=vals.shape[0])
            colors = X_test[:, feat_idx] if feat_idx < X_test.shape[1] else vals[:, feat_idx]
            ax.scatter(
                vals[:, feat_idx], y_positions,
                c=colors, cmap="coolwarm", s=10, alpha=0.7,
            )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]])
        ax.set_xlabel("SHAP value")
        ax.set_title("SHAP Beeswarm Plot")
        plt.tight_layout()
        beeswarm_path = output_dir / "shap_summary_beeswarm.png"
        fig.savefig(beeswarm_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(beeswarm_path)
        logger.info("  Saved %s", beeswarm_path)
    except Exception as e:
        logger.warning("  SHAP beeswarm plot failed: %s", e)

    # Waterfall plots for top N samples
    for sample_idx in range(min(max_waterfall_samples, vals.shape[0])):
        try:
            sample_vals = vals[sample_idx]
            sorted_idx = np.argsort(np.abs(sample_vals))[::-1][:10]

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in sample_vals[sorted_idx[::-1]]]
            ax.barh(
                range(len(sorted_idx)),
                sample_vals[sorted_idx[::-1]],
                color=colors,
            )
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]])
            ax.set_xlabel("SHAP value")
            ax.set_title(f"SHAP Waterfall — Sample {sample_idx}")
            plt.tight_layout()
            wf_path = output_dir / f"shap_waterfall_sample_{sample_idx}.png"
            fig.savefig(wf_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(wf_path)
        except Exception as e:
            logger.warning("  SHAP waterfall sample %d failed: %s", sample_idx, e)

    return saved


def generate_lime_plots(
    explanations: List[Dict[str, Any]],
    output_dir: Path,
    dpi: int = 300,
) -> List[Path]:
    """Generate LIME explanation bar charts.

    Args:
        explanations: List of LIME explanation dicts (feature_name → weight).
        output_dir: Directory to save plots.
        dpi: Figure resolution.

    Returns:
        List of saved plot paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping LIME plots")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for sample_idx, explanation in enumerate(explanations):
        try:
            if not explanation:
                continue

            features = list(explanation.keys())[:10]
            weights = [float(explanation[f]) for f in features]

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#d62728" if w > 0 else "#1f77b4" for w in weights[::-1]]
            ax.barh(range(len(features)), [weights[i] for i in range(len(features) - 1, -1, -1)], color=colors)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features[::-1])
            ax.set_xlabel("Feature weight")
            ax.set_title(f"LIME Explanation — Sample {sample_idx}")
            plt.tight_layout()
            lime_path = output_dir / f"lime_explanation_sample_{sample_idx}.png"
            fig.savefig(lime_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(lime_path)
        except Exception as e:
            logger.warning("  LIME plot sample %d failed: %s", sample_idx, e)

    return saved
