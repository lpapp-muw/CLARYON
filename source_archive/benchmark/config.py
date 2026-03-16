#!/usr/bin/env python3
"""
config.py — Benchmark harness configuration.

Central place for paths, dataset registry, seeds, ensemble size,
competitor list, and metric definitions.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────
# Override via environment variables on the benchmarking machine.
BASE_DIR = os.environ.get("MH_BENCH_BASE", "/home/morphedron/MH-Benchmark")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "benchmark_preprocessed")
RUNS_DIR = os.path.join(BASE_DIR, "benchmark_runs")
RESULTS_DIR = os.path.join(BASE_DIR, "benchmark_results")
DEBINN_BINARY = os.environ.get("MH_DEBINN_BIN", os.path.join(BASE_DIR, "DEBI", "build", "debinn"))

# ── CSV format (must match DEBI-NN C++ expectations) ──────────────────
CSV_SEP = ";"
FLOAT_FMT = "%.8f"

# ── Evaluation protocol ───────────────────────────────────────────────
CV_SEEDS = [42, 123, 456]
N_FOLDS = 5
TEST_RATIO = 0.2           # For k-fold: 1/N_FOLDS. For large datasets: explicit.
LARGE_DATASET_THRESHOLD = 10000   # N > this → fixed split instead of k-fold
LARGE_SPLIT_RATIOS = (0.6, 0.2, 0.2)  # train / val / test for large datasets

# ── Ensemble ──────────────────────────────────────────────────────────
ENSEMBLE_K = 1
ENSEMBLE_SEED_OFFSET = 1000  # Member i gets Optimizer/Seed = base_seed + ENSEMBLE_SEED_OFFSET + i

# ── Metrics (must match ResultLog.csv columns) ────────────────────────
PRIMARY_METRICS = ["BACC", "ACC", "EntropyLoss"]
ALL_METRICS = ["ACC", "BACC", "SNS", "SPC", "PPV", "NPV",
               "MCC", "MacroF1", "WeightedF1", "EntropyLoss"]

# ── 28 Benchmark Datasets ─────────────────────────────────────────────
# name → { n_samples, n_classes, source }
# n_samples is approximate — used only for large-dataset branching.
DATASETS = {
    # Tier 1: OpenML Standard (6)
    "australian":            {"n": 690,   "classes": 2, "source": "openml"},
    "blood-transfusion":     {"n": 748,   "classes": 2, "source": "openml"},
    "credit-g":              {"n": 1000,  "classes": 2, "source": "openml"},
    "diabetes":              {"n": 768,   "classes": 2, "source": "openml"},
    "kc1":                   {"n": 2109,  "classes": 2, "source": "openml"},
    "phoneme":               {"n": 5404,  "classes": 2, "source": "openml"},
    # Tier 2: Additional Standard (8)
    "iris":                  {"n": 150,   "classes": 3, "source": "openml"},
    "vehicle":               {"n": 846,   "classes": 4, "source": "openml"},
    "segment":               {"n": 2310,  "classes": 7, "source": "openml"},
    "waveform-5000":         {"n": 5000,  "classes": 3, "source": "openml"},
    "steel-plates-fault":    {"n": 1941,  "classes": 7, "source": "openml"},
    "electricity":           {"n": 45312, "classes": 2, "source": "openml"},
    "bank-marketing":        {"n": 45211, "classes": 2, "source": "openml"},
    "adult":                 {"n": 48842, "classes": 2, "source": "openml"},
    # Tier 3: Medical (8)
    "wisconsin-breast-cancer":{"n": 569,   "classes": 2, "source": "uci"},
    "heart-failure":         {"n": 299,   "classes": 2, "source": "kaggle"},
    "cervical-cancer":       {"n": 858,   "classes": 2, "source": "uci"},
    "chronic-kidney-disease": {"n": 400,   "classes": 2, "source": "uci"},
    "spect-heart":           {"n": 267,   "classes": 2, "source": "uci"},
    "hcc-survival":          {"n": 165,   "classes": 2, "source": "kaggle"},
    "mammographic-mass":     {"n": 961,   "classes": 2, "source": "uci"},
    "stroke-prediction":     {"n": 5110,  "classes": 2, "source": "kaggle"},
    # Tier 4: General Domain (6)
    "wine-quality":          {"n": 6497,  "classes": 7, "source": "uci"},
    "dry-bean":              {"n": 13611, "classes": 7, "source": "uci"},
    "drug-classification":   {"n": 200,   "classes": 5, "source": "kaggle"},
    "fetal-health":          {"n": 2126,  "classes": 3, "source": "kaggle"},
    "rice-cammeo-osmancik":   {"n": 3810,  "classes": 2, "source": "uci"},
    "mushroom":              {"n": 8124,  "classes": 2, "source": "uci"},
}

DATASET_NAMES = sorted(DATASETS.keys())

# Datasets using fixed split (N > LARGE_DATASET_THRESHOLD).
LARGE_DATASETS = [name for name, info in DATASETS.items()
                  if info["n"] > LARGE_DATASET_THRESHOLD]

# ── Competitors ────────────────────────────────────────────────────────
COMPETITORS = [
    "CatBoost",
    "XGBoost",
    "LightGBM",
    "TabPFN",
    "MLP",
    "TabM",
    "RealMLP",
    "ModernNCA",
]

# ── DEBI-NN runner ─────────────────────────────────────────────────────
DEBINN_TIMEOUT_SEC = 432000    # 5 days per project folder invocation
DEBINN_METHOD_NAMES = ["DEBINN-single", "DEBINN-ensemble"]
DEBINN_NUMA_NODES = 8        # Number of NUMA nodes (0 = disable numactl pinning)

# Auto-calculate OMP threads: all cores within one physical CPU socket.
# 256 cores / 8 NUMA nodes = 32 threads per node.
# Falls back to os.cpu_count() if NUMA is disabled.
if DEBINN_NUMA_NODES > 0:
    DEBINN_OMP_THREADS = max(1, os.cpu_count() // DEBINN_NUMA_NODES)
else:
    DEBINN_OMP_THREADS = min(32, os.cpu_count() or 4)
