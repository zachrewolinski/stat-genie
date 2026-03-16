#!/usr/bin/env python3
"""
Parallelized calibration simulation for blocking justification.

For each dataset, generates R known-null replicates (by permuting distribution
labels within perturbation blocks) and runs both blocked and unblocked
permutation tests. Results are saved to an .npz file for analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = EXPERIMENT_DIR / "aggregated_results" / "aggregated_results.csv"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "insights" / "calibration_results_{score}_{agg}.npz"

AGG_FNS: dict[str, Callable] = {"mean": np.mean, "median": np.median}


def blocked_perm_pvalue(
    df_sub: pd.DataFrame, score_col: str, B: int, agg: Callable, rng: np.random.Generator,
) -> float:
    """Blocked permutation test. Permutes distribution labels within each
    perturbation block; test statistic is the average within-block
    (alt agg - null agg) difference."""
    blocks = []
    for _, block in df_sub.groupby("perturbation"):
        alt_vals = block.loc[block["distribution"] == "alt", score_col].values
        null_vals = block.loc[block["distribution"] == "null", score_col].values
        if len(alt_vals) == 0 or len(null_vals) == 0:
            continue
        scores = np.concatenate([alt_vals, null_vals]).astype(float)
        blocks.append((scores, len(alt_vals)))

    def stat(blks):
        return np.mean([agg(s[:na]) - agg(s[na:]) for s, na in blks])

    t_obs = stat(blocks)
    count = 0
    for _ in range(B):
        for scores, _ in blocks:
            rng.shuffle(scores)
        if stat(blocks) >= t_obs:
            count += 1

    return (count + 1) / (B + 1)


def unblocked_perm_pvalue(
    y: np.ndarray, n_alt: int, B: int, agg: Callable, rng: np.random.Generator,
) -> float:
    """Unblocked permutation test. Shuffles all labels ignoring perturbation
    structure; test statistic is agg(alt) - agg(null)."""
    n = len(y)
    t_obs = float(agg(y[:n_alt]) - agg(y[n_alt:]))

    perms = np.argsort(rng.random((B, n)), axis=1)

    if agg is np.mean:
        # vectorized fast path
        perm_ts = (
            np.mean(y[perms[:, :n_alt]], axis=1)
            - np.mean(y[perms[:, n_alt:]], axis=1)
        )
    else:
        perm_ts = np.array([
            float(agg(y[p[:n_alt]]) - agg(y[p[n_alt:]]))
            for p in perms
        ])

    return ((perm_ts >= t_obs).sum() + 1) / (B + 1)


def run_one_replicate(
    dataset_name: str,
    sub_df: pd.DataFrame,
    score_col: str,
    rep_id: int,
    B: int,
    agg: Callable,
    child_seed: np.random.SeedSequence,
) -> tuple[str, int, float, float]:
    """Run blocked + unblocked tests on one known-null replicate."""
    rng = np.random.default_rng(child_seed)

    # construct known-null: permute distribution labels within each block
    null_sub = sub_df.copy()
    for _, idx in null_sub.groupby("perturbation").groups.items():
        labels = null_sub.loc[idx, "distribution"].values.copy()
        rng.shuffle(labels)
        null_sub.loc[idx, "distribution"] = labels

    p_blocked = blocked_perm_pvalue(null_sub, score_col, B, agg, rng)

    alt_mask = null_sub["distribution"].values == "alt"
    y = null_sub[score_col].values.astype(float)
    n_alt = alt_mask.sum()
    y_sorted = np.concatenate([y[alt_mask], y[~alt_mask]])
    p_unblocked = unblocked_perm_pvalue(y_sorted, n_alt, B, agg, rng)

    return (dataset_name, rep_id, p_blocked, p_unblocked)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallelized calibration simulation for blocking justification."
    )
    parser.add_argument("--R", type=int, default=1000, help="Replicates per dataset.")
    parser.add_argument("--B", type=int, default=2000, help="Permutations per test.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument(
        "--agg", choices=list(AGG_FNS), default="mean",
        help="Aggregation function for within-block summaries (default: mean).",
    )
    parser.add_argument(
        "--score-col",
        choices=["response", "confidence"],
        default="response",
        help="Which score column to use (default: response).",
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV, help="Path to aggregated_results.csv."
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .npz path (default: insights/calibration_results_{score}_{agg}.npz).",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Joblib parallelism (-1 = all cores)."
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(
            str(DEFAULT_OUTPUT).format(score=args.score_col, agg=args.agg)
        )

    agg_fn = AGG_FNS[args.agg]

    if not args.csv.exists():
        print(f"File not found: {args.csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv, keep_default_na=False)
    datasets = sorted(df["dataset"].unique())
    print(f"Datasets: {datasets}")
    print(f"R={args.R}, B={args.B}, agg={args.agg}, score={args.score_col}, seed={args.seed}")

    ss = np.random.SeedSequence(args.seed)
    n_tasks = len(datasets) * args.R
    child_seeds = ss.spawn(n_tasks)

    tasks = []
    for di, name in enumerate(datasets):
        sub = df[df["dataset"] == name].copy()
        for r in range(args.R):
            seed_idx = di * args.R + r
            tasks.append((name, sub, args.score_col, r, args.B, agg_fn, child_seeds[seed_idx]))

    print(f"Running {n_tasks} tasks with n_jobs={args.n_jobs} ...")
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_one_replicate)(*t) for t in tasks
    )

    pvals = {name: {"blocked": [], "unblocked": []} for name in datasets}
    for dataset_name, rep_id, p_b, p_u in results:
        pvals[dataset_name]["blocked"].append(p_b)
        pvals[dataset_name]["unblocked"].append(p_u)

    save_dict = {
        "datasets": np.array(datasets),
        "R": args.R, "B": args.B, "agg": args.agg,
        "score_col": args.score_col,
    }
    for name in datasets:
        save_dict[f"{name}_blocked"] = np.array(pvals[name]["blocked"])
        save_dict[f"{name}_unblocked"] = np.array(pvals[name]["unblocked"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **save_dict)
    print(f"Saved to {args.output}")

    alpha = 0.05
    for name in datasets:
        pb = np.array(pvals[name]["blocked"])
        pu = np.array(pvals[name]["unblocked"])
        print(
            f"  {name}: blocked={np.mean(pb <= alpha):.3f}"
            f"  unblocked={np.mean(pu <= alpha):.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
