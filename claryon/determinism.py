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

        Implementation note: Python ``random.seed`` and ``np.random.seed``
        are called LAST (after all optional library imports) because some
        third-party libraries (e.g. PennyLane >= 0.44) have import-time
        side effects that consume the global random state. If we seeded
        Python/NumPy first, those imports would silently invalidate the
        seed on the first call to this function (the imports are cached
        on subsequent calls, hiding the bug).
    """
    seed = int(seed)

    # Threading controls — environment only, no random state touched.
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

    # Trigger optional library imports FIRST so any import-time side effects
    # (e.g. consumption of Python's global random state) happen BEFORE we
    # set our seeds.
    try:
        import pennylane.numpy as _pnp  # type: ignore[import-untyped]  # noqa: F401
    except Exception:
        _pnp = None  # type: ignore[assignment]

    try:
        import torch as _torch
    except ImportError:
        _torch = None  # type: ignore[assignment]

    try:
        import numpy as _np
    except ImportError:
        _np = None  # type: ignore[assignment]

    # Now seed everything in order: Python first, then libraries.
    random.seed(seed)

    if _np is not None:
        _np.random.seed(seed)

    if _pnp is not None:
        try:
            _pnp.random.seed(seed)
        except Exception:
            pass

    if _torch is not None:
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
            _torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            _torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

    logger.debug("Determinism enforced: seed=%d, threads=%s", seed, threads)
