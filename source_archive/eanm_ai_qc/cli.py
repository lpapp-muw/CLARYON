from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .determinism import enforce_determinism
from .runner import RunConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qnm_qai")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--input", required=True, help="CSV file (tabular) or folder (NIfTI)")
    r.add_argument("--input-type", default="auto", choices=["auto", "tabular", "nifti"])
    r.add_argument("--infer", default=None, help="Optional inference input (CSV or folder). Labels optional.")
    r.add_argument("--methods", default="pl_kernel_svm,pl_qcnn_alt,pl_qcnn_muw")
    r.add_argument("--results-dir", default="Results")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--test-size", type=float, default=0.25)
    r.add_argument("--shots", type=int, default=None)

    # QCNN hyperparams (defaults tuned for tabular QCNN stability)
    r.add_argument("--qcnn-epochs", type=int, default=15)
    r.add_argument("--qcnn-lr", type=float, default=0.02)
    r.add_argument("--qcnn-batch-size", type=int, default=16)
    r.add_argument("--qcnn-init-scale", type=float, default=0.1)
    r.add_argument("--qcnn-dense-layers", type=int, default=1)
    r.add_argument("--qcnn-layers", type=int, default=None)

    # Runtime controls
    r.add_argument(
        "--max-samples-per-method",
        type=int,
        default=None,
        help="Subsample training rows per method (runtime control).",
    )
    r.add_argument(
        "--max-features-qcnn-tabular",
        type=int,
        default=None,
        help="Optional cap on raw feature count used for QCNN amplitude vectors (tabular only).",
    )

    # Explainability (SHAP/LIME)
    r.add_argument("--no-explain", action="store_true", help="Disable SHAP/LIME.")
    r.add_argument("--explain-max-features", type=int, default=32)
    r.add_argument("--explain-max-test-samples", type=int, default=5)
    r.add_argument("--explain-background-samples", type=int, default=12)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if args.cmd == "run":
        cfg = RunConfig(
            input_path=Path(args.input),
            input_type=str(args.input_type),
            infer_path=Path(args.infer) if args.infer else None,
            methods=[m.strip() for m in str(args.methods).split(",") if m.strip()],
            results_dir=Path(args.results_dir),
            seed=int(args.seed),
            test_size=float(args.test_size),
            shots=args.shots,

            qcnn_epochs=int(args.qcnn_epochs),
            qcnn_lr=float(args.qcnn_lr),
            qcnn_batch_size=args.qcnn_batch_size,
            qcnn_init_scale=float(args.qcnn_init_scale),
            qcnn_dense_layers=int(args.qcnn_dense_layers),
            qcnn_layers=args.qcnn_layers,

            max_samples_per_method=args.max_samples_per_method,
            max_features_qcnn_tabular=args.max_features_qcnn_tabular,

            explain=(not args.no_explain),
            explain_max_features=args.explain_max_features,
            explain_max_test_samples=args.explain_max_test_samples,
            explain_background_samples=args.explain_background_samples,
        )
        enforce_determinism(cfg.seed, threads=1)
        summary = run_experiment(cfg)
        print(f"[RESULTS] summary_csv={summary}")
        return
