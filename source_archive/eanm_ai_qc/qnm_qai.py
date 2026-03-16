#!/usr/bin/env python3
"""CLI entrypoint for EANM-AI-QC.

This entrypoint sets common BLAS thread pool variables to 1 thread (best-effort)
to reduce nondeterministic floating-point reductions and to keep CPU use predictable.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from eanm_ai_qc.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
