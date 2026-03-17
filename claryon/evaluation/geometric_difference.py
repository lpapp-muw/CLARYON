"""Geometric Difference framework (Huang et al. 2021) — quantum advantage analysis.

Implements the full quantum advantage assessment:
- Geometric difference g(K^C || K^Q)
- Model complexity s_K(N)
- Effective dimension d
- Decision logic (classical_sufficient / quantum_advantage_likely / inconclusive)

Reference: Huang et al., Nature Communications 12, 2631 (2021).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg.interpolative import estimate_spectral_norm
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

logger = logging.getLogger(__name__)


def geometric_difference_score(
    K_quantum: np.ndarray,
    K_classical: Optional[np.ndarray] = None,
    X_train: Optional[np.ndarray] = None,
    its: int = 20,
) -> float:
    """Compute g(K^C || K^Q) = sqrt(spectral_norm(K_Q^{1/2} @ K_C^{-1} @ K_Q^{1/2})).

    Args:
        K_quantum: Quantum kernel matrix, shape (N, N).
        K_classical: Classical kernel matrix, shape (N, N). If None, computed
            as linear kernel from X_train.
        X_train: Training features, used to compute K_classical if not provided.
        its: Number of iterations for spectral norm estimation.

    Returns:
        GDQ score (float). Values > 1 suggest quantum advantage.
    """
    if K_classical is None:
        if X_train is None:
            raise ValueError("Provide either K_classical or X_train")
        K_classical = linear_kernel(X_train, X_train)

    n = K_quantum.shape[0]
    reg = 1e-8 * np.eye(n)

    K_Q_sqrt = np.real(sqrtm(K_quantum + reg))
    K_C_inv = np.linalg.pinv(K_classical + reg)
    M = K_Q_sqrt @ K_C_inv @ K_Q_sqrt

    spectral_norm = estimate_spectral_norm(M, its=its)
    gdq = float(np.sqrt(max(spectral_norm, 0.0)))

    logger.info("GDQ score: %.4f", gdq)
    return gdq


def model_complexity(K: np.ndarray, y: np.ndarray) -> float:
    """Compute model complexity s_K(N) = y^T @ K^{-1} @ y.

    Args:
        K: Kernel matrix, shape (N, N).
        y: Labels, shape (N,).

    Returns:
        Model complexity score.
    """
    n = K.shape[0]
    K_inv = np.linalg.pinv(K + 1e-8 * np.eye(n))
    return float(y @ K_inv @ y)


def effective_dimension(K_Q: np.ndarray, threshold: float = 0.01) -> int:
    """Compute effective dimension d = rank(K_Q) with eigenvalue truncation.

    Args:
        K_Q: Quantum kernel matrix, shape (N, N).
        threshold: Fraction of max eigenvalue below which to truncate.

    Returns:
        Effective dimension.
    """
    eigenvalues = np.linalg.eigvalsh(K_Q)
    return int(np.sum(eigenvalues > threshold * eigenvalues.max()))


def quantum_advantage_analysis(
    K_Q: np.ndarray,
    y_train: np.ndarray,
    X_train: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run the full Huang et al. 2021 quantum advantage analysis.

    Tests the quantum kernel against linear, RBF, and polynomial classical kernels.

    Args:
        K_Q: Quantum kernel matrix, shape (N, N).
        y_train: Training labels, shape (N,).
        X_train: Training features (needed for classical kernel computation).

    Returns:
        Analysis results dict with g_CQ, s_C, s_Q, d, recommendation, explanation.
    """
    if X_train is None:
        # Use quantum kernel eigenspace as proxy
        X_train = np.real(sqrtm(K_Q + 1e-8 * np.eye(K_Q.shape[0])))

    # Convert labels to float for complexity computation
    y = y_train.astype(np.float64)
    if y.max() == y.min():
        y = np.ones_like(y, dtype=np.float64)

    # Classical kernels
    classical_kernels = {
        "linear": linear_kernel(X_train),
        "rbf": rbf_kernel(X_train, gamma=1.0 / X_train.shape[1]),
        "polynomial": polynomial_kernel(X_train, degree=3),
    }

    # Geometric difference per classical kernel
    g_CQ: Dict[str, float] = {}
    for name, K_C in classical_kernels.items():
        try:
            g_CQ[name] = geometric_difference_score(K_Q, K_C)
        except Exception as e:
            logger.warning("GDQ for %s failed: %s", name, e)
            g_CQ[name] = float("nan")

    # Model complexity
    s_C: Dict[str, float] = {}
    for name, K_C in classical_kernels.items():
        try:
            s_C[name] = model_complexity(K_C, y)
        except Exception as e:
            logger.warning("Complexity for %s failed: %s", name, e)
            s_C[name] = float("nan")

    s_Q = model_complexity(K_Q, y)

    # Effective dimension
    d = effective_dimension(K_Q)
    K_Q_rank = int(np.linalg.matrix_rank(K_Q + 1e-8 * np.eye(K_Q.shape[0])))

    # Decision logic (Huang et al. Figure 1)
    valid_g = [v for v in g_CQ.values() if v == v]  # filter NaN  # noqa: PLR0124
    max_g = max(valid_g) if valid_g else 0.0
    min_s_C = min((v for v in s_C.values() if v == v), default=float("inf"))  # noqa: PLR0124

    if max_g < 1.1:
        recommendation = "classical_sufficient"
        explanation = (
            f"Small geometric difference (max g={max_g:.2f}): "
            f"classical ML is guaranteed competitive."
        )
    elif s_Q < min_s_C * 0.5:
        recommendation = "quantum_advantage_likely"
        explanation = (
            f"Large g_CQ ({max_g:.2f}) and lower quantum complexity "
            f"(s_Q={s_Q:.2f} vs min s_C={min_s_C:.2f}) suggest advantage."
        )
    else:
        recommendation = "inconclusive"
        explanation = (
            f"Large geometric difference (max g={max_g:.2f}) but model "
            f"complexities are similar (s_Q={s_Q:.2f}, min s_C={min_s_C:.2f})."
        )

    return {
        "g_CQ": g_CQ,
        "s_C": s_C,
        "s_Q": s_Q,
        "d": d,
        "K_Q_rank": K_Q_rank,
        "recommendation": recommendation,
        "explanation": explanation,
    }


def generate_gdq_report(
    analysis: Dict[str, Any],
    output_dir: Path,
    dpi: int = 300,
) -> List[Path]:
    """Generate geometric difference visualization report.

    Args:
        analysis: Results from quantum_advantage_analysis().
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
        logger.warning("matplotlib not available — skipping GDQ plots")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: g_CQ bar chart
        g_CQ = analysis["g_CQ"]
        names = list(g_CQ.keys())
        values = [g_CQ[n] for n in names]
        colors = ["#d62728" if v > 1.1 else "#1f77b4" for v in values]
        axes[0].bar(names, values, color=colors)
        axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
        axes[0].set_ylabel("g(K^C || K^Q)")
        axes[0].set_title("Geometric Difference")

        # Panel 2: s_C vs s_Q
        s_C = analysis["s_C"]
        for name, val in s_C.items():
            axes[1].scatter([name], [val], label=f"classical ({name})", s=100)
        axes[1].axhline(y=analysis["s_Q"], color="#d62728", linestyle="--", label="quantum")
        axes[1].set_ylabel("Model complexity s_K(N)")
        axes[1].set_title("Model Complexity")
        axes[1].legend(fontsize=8)

        # Panel 3: Recommendation
        rec = analysis["recommendation"]
        color_map = {
            "classical_sufficient": "#1f77b4",
            "quantum_advantage_likely": "#2ca02c",
            "inconclusive": "#ff7f0e",
        }
        axes[2].text(
            0.5, 0.5, rec.replace("_", "\n"),
            ha="center", va="center",
            fontsize=16, fontweight="bold",
            color=color_map.get(rec, "black"),
            transform=axes[2].transAxes,
        )
        axes[2].text(
            0.5, 0.2, analysis["explanation"],
            ha="center", va="center",
            fontsize=9, wrap=True,
            transform=axes[2].transAxes,
        )
        axes[2].set_title("Recommendation")
        axes[2].axis("off")

        plt.tight_layout()
        path = output_dir / "geometric_difference_report.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info("Saved GDQ report to %s", path)
    except Exception as e:
        logger.warning("GDQ report generation failed: %s", e)

    return saved
