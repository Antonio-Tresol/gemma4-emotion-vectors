#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Falsification tests for Q1.H1: group B scores higher than group A.

Runs: descriptive stats, Welch t-test (via permutation), permutation null on
the mean difference, and a bootstrap CI on the mean difference. Prints a JSON
scorecard to stdout and results/falsification_scorecard.json.
"""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "pilot_data.csv"
OUT = ROOT / "results" / "falsification_scorecard.json"

SEED = 42
N_PERM = 100_000
N_BOOT = 100_000


def load():
    a, b = [], []
    with DATA.open() as f:
        for row in csv.DictReader(f):
            (a if row["group"] == "A" else b).append(float(row["score"]))
    return np.array(a), np.array(b)


def describe(x):
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def permutation_test(a, b, rng):
    observed = np.mean(b) - np.mean(a)
    pooled = np.concatenate([a, b])
    n_a = len(a)
    n = len(pooled)
    diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        rng.shuffle(pooled)
        diffs[i] = np.mean(pooled[n_a:]) - np.mean(pooled[:n_a])
    p_two_sided = float(np.mean(np.abs(diffs) >= abs(observed)))
    p_one_sided = float(np.mean(diffs >= observed))
    return observed, p_two_sided, p_one_sided, diffs


def bootstrap_ci(a, b, rng, alpha=0.05):
    boot_diffs = np.empty(N_BOOT)
    n_a, n_b = len(a), len(b)
    for i in range(N_BOOT):
        sa = a[rng.integers(0, n_a, n_a)]
        sb = b[rng.integers(0, n_b, n_b)]
        boot_diffs[i] = np.mean(sb) - np.mean(sa)
    lo, hi = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cohens_d(a, b):
    n_a, n_b = len(a), len(b)
    pooled_sd = np.sqrt(((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2))
    return float((np.mean(b) - np.mean(a)) / pooled_sd)


def main():
    a, b = load()
    rng = np.random.default_rng(SEED)

    desc_a, desc_b = describe(a), describe(b)
    observed, p_two, p_one, _ = permutation_test(a, b, rng)
    ci_lo, ci_hi = bootstrap_ci(a, b, rng)
    d = cohens_d(a, b)

    result = {
        "claim": "Q1.H1: Group B scores higher than group A",
        "seed": SEED,
        "n_permutations": N_PERM,
        "n_bootstrap": N_BOOT,
        "group_a": desc_a,
        "group_b": desc_b,
        "observed_mean_diff_B_minus_A": observed,
        "permutation_p_two_sided": p_two,
        "permutation_p_one_sided_B_gt_A": p_one,
        "bootstrap_95ci_mean_diff_B_minus_A": [ci_lo, ci_hi],
        "cohens_d": d,
    }

    print(json.dumps(result, indent=2))
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
