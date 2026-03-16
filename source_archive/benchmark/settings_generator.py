#!/usr/bin/env python3
"""
settings_generator.py — Generate executionSettings.csv for DEBI-NN ensemble runs.

Takes a base settings template (single-column executionSettings.csv) and
replicates it K times with different Optimizer/Seed and Validation/Seed
per ensemble member.

Output:
    executionSettings.csv with columns: Keys;M0;M1;M2;M3;M4
    (K=5 ensemble members, each with unique seeds)
"""

import os
import argparse

from config import (
    ENSEMBLE_K, ENSEMBLE_SEED_OFFSET, CSV_SEP, DEBINN_OMP_THREADS,
)


def parse_template(template_path):
    """Parse a single-column executionSettings.csv into ordered (key, value) pairs.

    Returns:
        list of (key, value) tuples preserving file order.
    """
    entries = []
    with open(template_path, "r") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(CSV_SEP)
            if len(parts) < 2:
                continue
            key = parts[0]
            value = parts[1]
            # Skip the header row if present.
            if line_num == 0 and key == "Keys":
                continue
            entries.append((key, value))
    return entries


def generate_ensemble_settings(template_path, output_path, dataset_name,
                               base_seed, k=None, n_classes=None):
    """Generate a K-column executionSettings.csv for ensemble members.

    Args:
        template_path: Path to single-column executionSettings.csv template.
        output_path:   Where to write the multi-column CSV.
        dataset_name:  Value for the Data key.
        base_seed:     Base seed for this CV seed. Members get
                       base_seed + ENSEMBLE_SEED_OFFSET + member_idx.
        k:             Ensemble size (default: ENSEMBLE_K from config).
        n_classes:     Number of output classes for this dataset. If provided,
                       overrides Model/Layers/Output/NeuronCount in template.
    """
    if k is None:
        k = ENSEMBLE_K

    entries = parse_template(template_path)

    # Build a dict for quick key lookup while preserving order.
    key_order = [e[0] for e in entries]
    values = {e[0]: e[1] for e in entries}

    # Column names.
    col_names = [f"M{i}" for i in range(k)]

    lines = []
    # Header row.
    lines.append(CSV_SEP.join(["Keys"] + col_names))

    for key in key_order:
        base_value = values[key]
        row_values = []

        for member_idx in range(k):
            member_seed = base_seed + ENSEMBLE_SEED_OFFSET + member_idx

            if key == "Data":
                row_values.append(dataset_name)
            elif key == "Optimizer/Seed":
                row_values.append(str(member_seed))
            elif key == "Validation/Seed":
                row_values.append(str(member_seed))
            elif key == "Model/Layers/Output/NeuronCount" and n_classes is not None:
                row_values.append(str(n_classes))
            elif key == "Optimizer/OMPThreads":
                row_values.append(str(DEBINN_OMP_THREADS))
            else:
                row_values.append(base_value)

        lines.append(CSV_SEP.join([key] + row_values))

    # Ensure Data key exists (add if template doesn't have it).
    if "Data" not in values:
        row_values = [dataset_name] * k
        lines.insert(1, CSV_SEP.join(["Data"] + row_values))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate ensemble executionSettings.csv from template")
    parser.add_argument("--template", required=True,
                        help="Path to base executionSettings.csv (single column)")
    parser.add_argument("--output", required=True,
                        help="Output path for multi-column CSV")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (Data key value)")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base seed for this run")
    parser.add_argument("--k", type=int, default=ENSEMBLE_K,
                        help=f"Ensemble member count (default: {ENSEMBLE_K})")
    parser.add_argument("--n-classes", type=int, default=None,
                        help="Number of output classes (overrides template Output/NeuronCount)")
    args = parser.parse_args()

    out = generate_ensemble_settings(
        args.template, args.output, args.dataset, args.base_seed, args.k,
        args.n_classes)
    print(f"Generated: {out} ({args.k} columns)")


if __name__ == "__main__":
    main()
