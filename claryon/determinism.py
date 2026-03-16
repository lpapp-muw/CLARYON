"""Determinism controls: seed all RNGs + limit BLAS threads."""
from __future__ import annotations

import logging
import os
import random
from typing import Optional

logger = logging.getLogger(__name__)


def enforce_determinism(seed: int, threads: Optional[int] = 1) -> None:
    """Best-effort determinism controls for reproducible experiments.

    Seeds Python, NumPy, PennyLane, and PyTorch RNGs. Limits BLAS thread
    pools to reduce nondeterministic float reductions.

    Args:
        seed: Master seed for all RNGs.
        threads: Number of threads for BLAS libraries. Set to 1 for maximum
            reproducibility. ``None`` leaves threading untouched.

    Note:
        True bit-for-bit determinism across machines/BLAS builds is not
        guaranteed. This function makes runs repeatable on the same machine.
        For best results, call before importing numpy/scipy/sklearn.
    """
    seed = int(seed)

    # Threading controls
    if threads is not None:
        t = str(int(threads))
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(var, t)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PennyLane numpy (autograd wrapper)
    try:
        import pennylane.numpy as pnp  # type: ignore[import-untyped]
        pnp.random.seed(seed)
    except Exception:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except ImportError:
        pass

    logger.debug("Determinism enforced: seed=%d, threads=%s", seed, threads)
