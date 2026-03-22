#!/usr/bin/env python3
"""
KDE plots of agent response scores under null vs alternative distributions,
one panel per BLADE dataset, ordered by permutation-test p-value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = EXPERIMENT_DIR / "aggregated_results" / "aggregated_results.csv"
FIGURES_DIR = EXPERIMENT_DIR / "insights" / "figures"

def diff_in_means_perm_test(alt, null, n_perm=10_000, rng=None):
    """Unblocked difference-in-means permutation test."""
    if rng is None:
        rng = np.random.default_rng()
    n_alt = len(alt)
    pooled = np.concatenate([alt, null]).astype(float)
    t_obs = float(pooled[:n_alt].mean() - pooled[n_alt:].mean())

    n = len(pooled)
    perms = np.argsort(rng.random((n_perm, n)), axis=1)
    perm_ts = (
        np.mean(pooled[perms[:, :n_alt]], axis=1)
        - np.mean(pooled[perms[:, n_alt:]], axis=1)
    )

    p_value = ((perm_ts >= t_obs).sum() + 1) / (n_perm + 1)
    return t_obs, p_value, perm_ts


def reflected_kde(vals, ax, color, label=None, lo=0, hi=100, n_pts=512,
                  bw_adjust=0.18):
    """KDE with boundary reflection on [lo, hi]."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return
    reflected = np.concatenate([v, 2 * lo - v, 2 * hi - v])
    kde = gaussian_kde(reflected)
    kde.set_bandwidth(kde.factor * bw_adjust)
    x = np.linspace(lo, hi, n_pts)
    y = kde(x) * 3
    ax.fill_between(x, y, alpha=0.3, color=color, label=label)
    ax.plot(x, y, color=color, linewidth=1.5)
    ax.axvline(np.mean(v), color=color, linestyle="--", linewidth=0.8)


def main() -> int:
    if not DEFAULT_CSV.exists():
        print(f"File not found: {DEFAULT_CSV}", file=sys.stderr)
        return 1

    df = pd.read_csv(DEFAULT_CSV, keep_default_na=False)
    datasets = sorted(df["dataset"].unique())

    # compute per-dataset p-values (unblocked difference in means)
    rng = np.random.default_rng(42)
    pvals = {}
    for name in datasets:
        sub = df[df["dataset"] == name]
        alt_v = sub.loc[sub["distribution"] == "alt", "response"].values
        null_v = sub.loc[sub["distribution"] == "null", "response"].values
        if len(alt_v) >= 2 and len(null_v) >= 2:
            _, p, _ = diff_in_means_perm_test(alt_v, null_v, n_perm=2_000, rng=rng)
        else:
            p = 1.0
        pvals[name] = p

    # sort datasets by p-value
    ordered = sorted(datasets, key=lambda d: pvals[d])

    n_datasets = len(ordered)
    n_cols = 2
    n_rows = int(np.ceil(n_datasets / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7, 2.6 * n_rows),
        sharey=False,
    )
    axes = axes.flatten()

    for i, name in enumerate(ordered):
        ax = axes[i]
        sub = df[df["dataset"] == name]
        null_vals = sub.loc[sub["distribution"] == "null", "response"].values
        alt_vals = sub.loc[sub["distribution"] == "alt", "response"].values

        reflected_kde(null_vals, ax, color="#4C72B0", label="Null")
        reflected_kde(alt_vals, ax, color="#DD8452", label="Alt")

        null_mean = null_vals.mean()
        alt_mean = alt_vals.mean()
        p = pvals[name]
        p_str = f"$p < 0.001$" if p < 0.001 else f"$p = {p:.3f}$"
        ax.set_title(
            f"{name}  ({p_str})  means=[{null_mean:.0f}, {alt_mean:.0f}]",
            fontsize=10,
        )
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=8)

    # hide unused panels
    for j in range(n_datasets, len(axes)):
        axes[j].set_visible(False)

    # axis labels on bottom row only
    for j in range(n_cols):
        idx = (n_rows - 1) * n_cols + j
        if idx < len(axes) and axes[idx].get_visible():
            axes[idx].set_xlabel("Response (0–100)")

    fig.suptitle(
        "Response score distributions by dataset (ordered by $p$-value)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "response_kde_by_dataset.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
