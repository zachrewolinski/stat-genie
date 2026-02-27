#!/usr/bin/env python3
"""
Blocked stratified permutation test for scalar experiment results.

Shuffles alt/null condition labels within each perturbation block,
computes the average of per-block differences (median or mean),
and reports one-sided p-values per dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = EXPERIMENT_DIR / "aggregated_results.csv"


def stratified_diff(blocks: list[tuple[np.ndarray, int]], agg=np.median) -> float:
    """Average of within-block (alt - null) differences using `agg`."""
    diffs = []
    for scores, n_alt in blocks:
        diffs.append(float(agg(scores[:n_alt]) - agg(scores[n_alt:])))
    return np.mean(diffs)


def blocked_permutation_test(
    sub: pd.DataFrame,
    n_perm: int = 10_000,
    agg=np.median,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, np.ndarray]:
    """
    Stratified permutation test: shuffles alt/null labels within each
    perturbation block, computes the average of per-block differences
    (using `agg`), repeats.

    Returns (observed_T, p_value, permuted_T_array).
    """
    if rng is None:
        rng = np.random.default_rng()

    # only keep blocks that have both alt and null observations
    block_counts = (
        sub.groupby(["perturbation", "distribution"]).size().unstack(fill_value=0)
    )
    valid_blocks = block_counts.index[
        (block_counts["alt"] > 0) & (block_counts["null"] > 0)
    ]
    sub = sub[sub["perturbation"].isin(valid_blocks)].copy()

    # precompute per-block arrays ordered as [alt scores, null scores]
    blocks = []
    for _, block in sub.groupby("perturbation"):
        alt_vals = block.loc[block["distribution"] == "alt", "response"].values
        null_vals = block.loc[block["distribution"] == "null", "response"].values
        scores = np.concatenate([alt_vals, null_vals])
        blocks.append((scores, len(alt_vals)))

    t_obs = stratified_diff(blocks, agg)

    perm_ts = np.empty(n_perm)
    for i in range(n_perm):
        for scores, _ in blocks:
            rng.shuffle(scores)
        perm_ts[i] = stratified_diff(blocks, agg)

    # Phipson & Smyth (2010) small-sample correction
    p_value = ((perm_ts >= t_obs).sum() + 1) / (n_perm + 1)
    return t_obs, p_value, perm_ts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run blocked stratified permutation tests on scalar experiment results."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to aggregated_results.csv.",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=10_000,
        help="Number of permutations (default: 10000).",
    )
    parser.add_argument(
        "--agg",
        choices=["median", "mean"],
        default="median",
        help="Aggregation function for within-block differences (default: median).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"File not found: {args.csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv, keep_default_na=False)
    datasets = sorted(df["dataset"].unique())
    agg_fn = np.median if args.agg == "median" else np.mean
    rng = np.random.default_rng(args.seed)

    results = []
    for name in datasets:
        sub = df[df["dataset"] == name]
        t_obs, p_val, _ = blocked_permutation_test(sub, args.n_perm, agg_fn, rng)
        results.append({"dataset": name, "T_obs": t_obs, "p_value": p_val})
        print(f"{name}: T={t_obs:.2f}  p={p_val:.4f}")

    print()
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
