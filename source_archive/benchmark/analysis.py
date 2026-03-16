#!/usr/bin/env python3
"""
analysis.py — Statistical analysis and visualization of benchmark results.

Reads results_table.csv and produces:
    - Mean rank table across datasets
    - Per-tier rank tables (small/medium/large)
    - Critical difference diagrams (Friedman + Nemenyi post-hoc)
    - Per-dataset comparison table
    - LaTeX-formatted tables for publication
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

from config import (
    RESULTS_DIR, DATASETS, LARGE_DATASET_THRESHOLD,
    PRIMARY_METRICS, DEBINN_METHOD_NAMES,
)


# ── Dataset size tiers (from E8 protocol) ──────────────────────────────

TIER_BOUNDS = {
    "Small":  (0, 500),
    "Medium": (500, 5000),
    "Large":  (5000, 1e9),
}


def get_tier(dataset_name):
    """Return size tier for a dataset."""
    n = DATASETS.get(dataset_name, {}).get("n", 0)
    for tier, (lo, hi) in TIER_BOUNDS.items():
        if lo <= n < hi:
            return tier
    return "Unknown"


def load_results(results_path):
    """Load results_table.csv."""
    return pd.read_csv(results_path)


def compute_mean_per_dataset(df, metric="BACC"):
    """Compute mean metric per dataset per method (across seeds and folds)."""
    grouped = df.groupby(["dataset", "method"])[metric].mean().reset_index()
    pivot = grouped.pivot(index="dataset", columns="method", values=metric)
    return pivot


def compute_ranks(pivot_df, higher_is_better=True):
    """Rank methods per dataset. Returns DataFrame of ranks."""
    if higher_is_better:
        ranks = pivot_df.rank(axis=1, ascending=False, method="average")
    else:
        ranks = pivot_df.rank(axis=1, ascending=True, method="average")
    return ranks


def friedman_nemenyi(ranks_df, alpha=0.05):
    """Friedman test + Nemenyi critical difference.

    Returns:
        friedman_stat, friedman_p, critical_difference
    """
    k = ranks_df.shape[1]  # number of methods
    n = ranks_df.shape[0]  # number of datasets

    # Friedman chi-squared.
    mean_ranks = ranks_df.mean()
    chi2 = (12 * n / (k * (k + 1))) * \
           ((mean_ranks ** 2).sum() - (k * (k + 1) ** 2) / 4)

    # F-distributed Friedman statistic (Iman-Davenport).
    ff = ((n - 1) * chi2) / (n * (k - 1) - chi2) if (n * (k - 1) - chi2) > 0 else np.inf
    p_value = 1.0 - stats.f.cdf(ff, k - 1, (k - 1) * (n - 1))

    # Nemenyi critical difference.
    # q_alpha values for Nemenyi test (two-tailed, alpha=0.05).
    # Approximation: q_alpha ≈ stats.studentized_range.ppf(1-alpha, k, inf) / sqrt(2)
    # For practical k values, use tabulated values.
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha = q_alpha_table.get(k, 3.0)  # fallback
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    return chi2, p_value, cd


def generate_rank_table(df, metric="BACC", datasets=None):
    """Generate a rank summary table for a set of datasets."""
    if datasets is not None:
        df = df[df["dataset"].isin(datasets)]

    higher_is_better = metric != "EntropyLoss"
    pivot = compute_mean_per_dataset(df, metric)
    ranks = compute_ranks(pivot, higher_is_better)

    mean_ranks = ranks.mean().sort_values()
    mean_scores = pivot.mean().reindex(mean_ranks.index)

    summary = pd.DataFrame({
        "Mean Rank": mean_ranks,
        f"Mean {metric}": mean_scores,
    })

    return summary, ranks


def generate_per_dataset_table(df, metric="BACC"):
    """Generate per-dataset metric table (datasets × methods)."""
    # Mean ± std across seeds/folds.
    grouped = df.groupby(["dataset", "method"])[metric]
    mean_df = grouped.mean().reset_index().pivot(
        index="dataset", columns="method", values=metric)
    std_df = grouped.std().reset_index().pivot(
        index="dataset", columns="method", values=metric)

    return mean_df, std_df


def format_latex_table(mean_df, std_df, metric="BACC", bold_best=True):
    """Format a mean±std table as LaTeX."""
    lines = []
    methods = list(mean_df.columns)

    # Header.
    header = "Dataset & " + " & ".join(methods) + " \\\\"
    lines.append("\\begin{tabular}{l" + "c" * len(methods) + "}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")

    higher_is_better = metric != "EntropyLoss"

    for ds in mean_df.index:
        row_means = mean_df.loc[ds]
        row_stds = std_df.loc[ds]

        if higher_is_better:
            best_val = row_means.max()
        else:
            best_val = row_means.min()

        cells = [ds.replace("_", "\\_")]
        for m in methods:
            val = row_means.get(m, np.nan)
            std = row_stds.get(m, np.nan)
            if np.isnan(val):
                cells.append("---")
            else:
                cell = f"{val:.3f}$\\pm${std:.3f}" if not np.isnan(std) else f"{val:.3f}"
                if bold_best and abs(val - best_val) < 1e-6:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)

        lines.append(" & ".join(cells) + " \\\\")

    # Mean rank row.
    lines.append("\\midrule")
    pivot = compute_mean_per_dataset(
        pd.DataFrame({
            "dataset": [ds for ds in mean_df.index for m in methods],
            "method": methods * len(mean_df.index),
            metric: [mean_df.loc[ds].get(m, np.nan)
                     for ds in mean_df.index for m in methods],
        }), metric)
    ranks = compute_ranks(pivot, higher_is_better)
    mean_ranks = ranks.mean()

    rank_cells = ["\\textit{Mean Rank}"]
    best_rank = mean_ranks.min()
    for m in methods:
        r = mean_ranks.get(m, np.nan)
        cell = f"{r:.2f}"
        if abs(r - best_rank) < 1e-6:
            cell = f"\\textbf{{{cell}}}"
        rank_cells.append(cell)
    lines.append(" & ".join(rank_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results")
    parser.add_argument("--results", default=None,
                        help="Path to results_table.csv")
    parser.add_argument("--metric", default="BACC",
                        help="Primary metric for ranking (default: BACC)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for analysis artifacts")
    args = parser.parse_args()

    results_path = args.results or os.path.join(RESULTS_DIR, "results_table.csv")
    output_dir = args.output_dir or os.path.join(RESULTS_DIR, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    df = load_results(results_path)
    metric = args.metric
    higher_is_better = metric != "EntropyLoss"

    print(f"Analysis | metric={metric} | {len(df)} rows")
    print(f"  Methods: {sorted(df['method'].unique())}")
    print(f"  Datasets: {df['dataset'].nunique()}")
    print(f"{'=' * 60}")

    # ── Overall rank table ────────────────────────────────────────
    summary, ranks = generate_rank_table(df, metric)
    print(f"\nOverall Mean Rank ({metric}):")
    print(summary.to_string())
    summary.to_csv(os.path.join(output_dir, f"rank_table_{metric}.csv"))

    # ── Friedman + Nemenyi ────────────────────────────────────────
    if ranks.shape[0] >= 3 and ranks.shape[1] >= 2:
        chi2, p_val, cd = friedman_nemenyi(ranks)
        print(f"\nFriedman test: chi2={chi2:.2f}, p={p_val:.4f}")
        print(f"Nemenyi CD (alpha=0.05): {cd:.3f}")

        with open(os.path.join(output_dir, "friedman_nemenyi.txt"), "w") as f:
            f.write(f"Metric: {metric}\n")
            f.write(f"Friedman chi2: {chi2:.4f}\n")
            f.write(f"Friedman p-value: {p_val:.6f}\n")
            f.write(f"Nemenyi CD (alpha=0.05): {cd:.4f}\n")
            f.write(f"\nMean ranks:\n{ranks.mean().sort_values().to_string()}\n")

    # ── Per-tier rank tables ──────────────────────────────────────
    df["tier"] = df["dataset"].map(get_tier)
    for tier in ["Small", "Medium", "Large"]:
        tier_datasets = [ds for ds in DATASETS if get_tier(ds) == tier]
        if not tier_datasets:
            continue

        tier_df = df[df["dataset"].isin(tier_datasets)]
        if tier_df.empty:
            continue

        tier_summary, _ = generate_rank_table(tier_df, metric, tier_datasets)
        print(f"\n{tier} datasets rank ({metric}):")
        print(tier_summary.to_string())
        tier_summary.to_csv(
            os.path.join(output_dir, f"rank_table_{metric}_{tier.lower()}.csv"))

    # ── Per-dataset table ─────────────────────────────────────────
    mean_df, std_df = generate_per_dataset_table(df, metric)
    mean_df.to_csv(os.path.join(output_dir, f"per_dataset_{metric}_mean.csv"))
    std_df.to_csv(os.path.join(output_dir, f"per_dataset_{metric}_std.csv"))

    # ── LaTeX table ───────────────────────────────────────────────
    latex = format_latex_table(mean_df, std_df, metric)
    latex_path = os.path.join(output_dir, f"table_{metric}.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table: {latex_path}")

    print(f"\nAnalysis artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
