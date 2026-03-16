from __future__ import annotations

import os
import random
from typing import Optional

def enforce_determinism(seed: int, threads: int = 1) -> None:
    """Best-effort determinism controls.

    Notes:
    - True bit-for-bit determinism across machines/BLAS builds is not guaranteed.
    - This function makes runs *repeatable on the same machine* by:
      - seeding Python + NumPy RNGs
      - limiting common BLAS thread pools to 1 thread (reduces nondeterministic reductions)
    """
    seed = int(seed)

    # Threading controls (best set before importing numpy/scipy/sklearn; we still set as best-effort here)
    if threads is not None:
        t = str(int(threads))
        os.environ.setdefault("OMP_NUM_THREADS", t)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", t)
        os.environ.setdefault("MKL_NUM_THREADS", t)
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", t)
        os.environ.setdefault("NUMEXPR_NUM_THREADS", t)

    # Hash seed: only fully effective if set before interpreter start, but we still set for visibility
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)

    try:
        import numpy as np  # noqa: F401
        np.random.seed(seed)  # type: ignore[name-defined]
    except Exception:
        pass

    # PennyLane's wrapped numpy (autograd) can be seeded separately, but our weight init uses NumPy.
    try:
        import pennylane.numpy as pnp  # type: ignore
        pnp.random.seed(seed)
    except Exception:
        pass
