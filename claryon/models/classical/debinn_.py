"""DEBI-NN C++ subprocess wrapper — ModelBuilder interface for the external binary.

Ported from [B] debinn_runner.py. The DEBI-NN binary is invoked via subprocess.
It reads executionSettings.csv and writes Predictions.csv into Executions-Finished/.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...io.base import TaskType
from ...io.predictions import SEP, read_predictions
from ...registry import register
from ..base import InputType, ModelBuilder

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 432000  # 5 days
DEFAULT_OMP_THREADS = 4


@register("model", "debinn")
class DEBINNModel(ModelBuilder):
    """DEBI-NN deep ensemble binary via C++ subprocess.

    The model wraps the external DEBI-NN C++ binary. Training and prediction
    happen together in a single invocation: the binary reads a project folder,
    performs training, and writes predictions.

    Args:
        binary_path: Path to the DEBI-NN executable.
        timeout: Subprocess timeout in seconds.
        omp_threads: Number of OMP threads for BLAS.
        numa_node: NUMA node to pin to (None = no pinning).
    """

    def __init__(
        self,
        binary_path: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        omp_threads: int = DEFAULT_OMP_THREADS,
        numa_node: Optional[int] = None,
        **params: Any,
    ) -> None:
        self._binary_path = binary_path
        self._timeout = timeout
        self._omp_threads = omp_threads
        self._numa_node = numa_node
        self._params = params
        self._predictions: Optional[pd.DataFrame] = None
        self._project_dir: Optional[Path] = None

    @property
    def name(self) -> str:
        return "debinn"

    @property
    def input_type(self) -> InputType:
        return InputType.TABULAR

    @property
    def supports_tasks(self) -> tuple[TaskType, ...]:
        return (TaskType.BINARY, TaskType.MULTICLASS)

    def fit(
        self, X: np.ndarray, y: np.ndarray, task_type: TaskType,
        project_dir: Optional[str] = None, **kwargs: Any,
    ) -> None:
        """Run DEBI-NN on the given project folder.

        DEBI-NN expects a pre-built project folder with executionSettings.csv,
        FDB/LDB files, etc. The ``project_dir`` kwarg must point to this folder.

        Args:
            X: Ignored (data is read from project folder by DEBI-NN).
            y: Ignored.
            task_type: Task type (must be classification).
            project_dir: Path to DEBI-NN project folder. Required.
        """
        if project_dir is None:
            raise ValueError("DEBI-NN requires project_dir kwarg")

        self._project_dir = Path(project_dir)
        result = self._invoke_binary(self._project_dir)

        if not result["success"]:
            raise RuntimeError(
                f"DEBI-NN failed (rc={result['returncode']}): {result['stderr'][:500]}"
            )

        logger.info("DEBI-NN completed in %.1fs", result["elapsed_sec"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions from the last DEBI-NN run."""
        if self._predictions is None:
            raise RuntimeError("No predictions available. Call fit() first.")
        return self._predictions["Predicted"].values.astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix from the last DEBI-NN run."""
        if self._predictions is None:
            raise RuntimeError("No predictions available. Call fit() first.")
        prob_cols = sorted(
            [c for c in self._predictions.columns if c.startswith("P") and c[1:].isdigit()],
            key=lambda c: int(c[1:]),
        )
        return self._predictions[prob_cols].values.astype(np.float64)

    def save(self, model_dir: Path) -> None:
        """Save project dir path reference."""
        model_dir.mkdir(parents=True, exist_ok=True)
        if self._project_dir:
            (model_dir / "project_dir.txt").write_text(str(self._project_dir))

    def load(self, model_dir: Path) -> None:
        """Load project dir path reference."""
        ref = model_dir / "project_dir.txt"
        if ref.exists():
            self._project_dir = Path(ref.read_text().strip())

    def load_predictions(self, pred_path: Path) -> None:
        """Manually load predictions from a Predictions.csv file.

        Args:
            pred_path: Path to Predictions.csv.
        """
        self._predictions = read_predictions(pred_path)

    def _invoke_binary(self, project_dir: Path) -> Dict[str, Any]:
        """Invoke the DEBI-NN binary on a project folder.

        Args:
            project_dir: Path to project folder.

        Returns:
            Result dict with success, returncode, elapsed_sec, stdout, stderr.
        """
        binary = self._binary_path
        if binary is None or not Path(binary).exists():
            return {
                "success": False,
                "returncode": -1,
                "elapsed_sec": 0.0,
                "stdout": "",
                "stderr": f"DEBI-NN binary not found: {binary}",
            }

        proj = str(project_dir).rstrip("/") + "/"

        cmd: List[str] = []
        if self._numa_node is not None:
            cmd = [
                "numactl",
                f"--cpunodebind={self._numa_node}",
                f"--membind={self._numa_node}",
            ]
        cmd.extend([binary, proj])

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self._omp_threads)
        env["QT_QPA_PLATFORM"] = "offscreen"

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(project_dir),
                env=env,
            )
            elapsed = time.time() - t0
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "elapsed_sec": elapsed,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -999,
                "elapsed_sec": time.time() - t0,
                "stdout": "",
                "stderr": f"TIMEOUT after {self._timeout}s",
            }
