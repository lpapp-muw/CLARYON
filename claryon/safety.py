"""Resource estimation and OOM protection for CLARYON pipeline."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .models.auto_complexity import estimate_runtime

logger = logging.getLogger(__name__)


def get_available_memory_gb() -> float:
    """Get available system memory in GB.

    Returns:
        Available memory in GB, or 8.0 as fallback.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass

    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass

    return 8.0  # conservative fallback


def estimate_memory_gb(
    model_name: str,
    n_qubits: int,
    n_samples: int,
) -> float:
    """Estimate memory usage in GB.

    Args:
        model_name: Model registry name.
        n_qubits: Number of qubits.
        n_samples: Number of training samples.

    Returns:
        Estimated memory in GB.
    """
    total = 0.0

    # State vector memory: 2^n_qubits complex numbers = 16 bytes each
    state_vector_bytes = 16 * (2 ** n_qubits)
    total += state_vector_bytes / 1e9

    # Kernel matrix for kernel-based models
    if model_name in ("kernel_svm", "sq_kernel_svm", "qdc_hadamard", "qdc_swap", "quantum_gp"):
        kernel_bytes = n_samples ** 2 * 8  # float64
        total += kernel_bytes / 1e9

    # SWAP test uses 2n+1 qubits
    if model_name == "qdc_swap":
        swap_qubits = 2 * n_qubits + 1
        swap_bytes = 16 * (2 ** swap_qubits)
        total += swap_bytes / 1e9

    return total


def preflight_resource_check(
    model_name: str,
    model_type: str,
    n_samples: int,
    n_qubits: int,
    params: Dict[str, Any],
) -> List[str]:
    """Check for potential resource issues before training.

    Args:
        model_name: Model registry name.
        model_type: Model category.
        n_samples: Number of training samples.
        n_qubits: Number of qubits.
        params: Resolved hyperparameters.

    Returns:
        List of warning messages.
    """
    warnings: List[str] = []

    # Memory: quantum state vector
    state_vector_bytes = 16 * (2 ** n_qubits)
    if state_vector_bytes > 1e9:
        warnings.append(
            f"MEMORY WARNING: {model_name} requires {state_vector_bytes / 1e9:.1f} GB "
            f"for state vector alone ({n_qubits} qubits). "
            f"Consider reducing features via mRMR or max_features."
        )

    # Memory: kernel matrix
    if model_name in ("kernel_svm", "sq_kernel_svm", "qdc_hadamard", "qdc_swap", "quantum_gp"):
        kernel_bytes = n_samples ** 2 * 8
        if kernel_bytes > 2e9:
            warnings.append(
                f"MEMORY WARNING: {model_name} kernel matrix needs "
                f"{kernel_bytes / 1e9:.1f} GB ({n_samples}\u00b2 entries). "
                f"Consider subsampling or using a training-based quantum model."
            )

    # SWAP test: 2n+1 qubits
    if model_name == "qdc_swap":
        swap_qubits = 2 * n_qubits + 1
        swap_bytes = 16 * (2 ** swap_qubits)
        if swap_bytes > 1e9:
            warnings.append(
                f"MEMORY WARNING: qdc_swap uses {swap_qubits} qubits "
                f"({swap_bytes / 1e9:.1f} GB state vector). "
                f"Consider qdc_hadamard ({n_qubits + 1} qubits) instead."
            )

    # Qubit count warnings
    if n_qubits > 20:
        warnings.append(
            f"RUNTIME WARNING: {n_qubits} qubits \u2014 simulation cost O(2^{n_qubits}). "
            f"Estimated memory: {state_vector_bytes / 1e9:.1f} GB. "
            f"This WILL be extremely slow. Reduce features."
        )
    elif n_qubits > 15:
        warnings.append(
            f"RUNTIME WARNING: {n_qubits} qubits \u2014 expect long runtimes. "
            f"Consider setting max_features to limit qubit count."
        )

    # Runtime estimate
    estimated_seconds = estimate_runtime(model_name, model_type, n_samples, n_qubits, params)
    if estimated_seconds > 3600:
        warnings.append(
            f"RUNTIME WARNING: {model_name} estimated at {estimated_seconds / 3600:.1f} hours per fold."
        )

    return warnings
