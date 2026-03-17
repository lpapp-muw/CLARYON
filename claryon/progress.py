"""CLI progress formatting and summary table."""
from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Stage markers
_CHECK = "\u2713"  # ✓


class ProgressDisplay:
    """Stage-level progress display for CLI output.

    Writes progress to stderr so stdout can be piped.
    Respects verbosity: 0=summary only, 1=stages+summary, 2=stages+details+summary.
    """

    def __init__(self, verbosity: int = 1, n_stages: int = 8) -> None:
        self.verbosity = verbosity
        self.n_stages = n_stages
        self._stage_idx = 0
        self._t_start = time.monotonic()

    def stage_start(self, name: str) -> None:
        """Mark the start of a pipeline stage."""
        self._stage_idx += 1
        if self.verbosity >= 1:
            sys.stderr.write(f"[{self._stage_idx}/{self.n_stages}] {name}...")
            sys.stderr.flush()

    def stage_done(self, detail: str = "") -> None:
        """Mark the current stage as complete."""
        if self.verbosity >= 1:
            suffix = f"  {_CHECK} {detail}" if detail else f"  {_CHECK}"
            sys.stderr.write(f"{suffix}\n")
            sys.stderr.flush()

    def model_progress(self, model_name: str, fold: int, total: int, elapsed: float) -> None:
        """Update per-model fold progress (verbosity >= 2)."""
        if self.verbosity >= 2:
            bar_len = 20
            filled = int(bar_len * fold / max(total, 1))
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
            sys.stderr.write(
                f"\r      {model_name:<14s} {bar} {fold:>2d}/{total} folds  "
                f"[{_fmt_time(elapsed)}]"
            )
            sys.stderr.flush()
            if fold >= total:
                sys.stderr.write("\n")
                sys.stderr.flush()

    def summary_table(
        self,
        metrics_summary: Dict[str, Dict[str, float]],
        metric_names: List[str],
        results_dir: str,
    ) -> None:
        """Print a summary table at the end of the pipeline."""
        if not metrics_summary:
            return

        try:
            from tabulate import tabulate
            headers = ["Model"] + [m.upper() for m in metric_names]
            rows = []
            for model_name, metrics in metrics_summary.items():
                row = [model_name]
                for m in metric_names:
                    val = metrics.get(m, float("nan"))
                    row.append(f"{val:.4f}" if val == val else "N/A")  # noqa: PLR0124
                rows.append(row)
            table = tabulate(rows, headers=headers, tablefmt="fancy_outline")
        except ImportError:
            # Fallback: simple format
            lines = []
            header = f"{'Model':<16s}" + "".join(f"{m:>12s}" for m in metric_names)
            lines.append(header)
            lines.append("-" * len(header))
            for model_name, metrics in metrics_summary.items():
                row = f"{model_name:<16s}"
                for m in metric_names:
                    val = metrics.get(m, float("nan"))
                    row += f"{val:>12.4f}" if val == val else f"{'N/A':>12s}"  # noqa: PLR0124
                lines.append(row)
            table = "\n".join(lines)

        elapsed = time.monotonic() - self._t_start
        sys.stderr.write(f"\n{table}\n\n")
        sys.stderr.write(f"Results saved to: {results_dir}/\n")
        sys.stderr.write(f"Runtime: {_fmt_time(elapsed)}\n")
        sys.stderr.flush()


def _fmt_time(seconds: float) -> str:
    """Format seconds as mm:ss or Xm Ys."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"
