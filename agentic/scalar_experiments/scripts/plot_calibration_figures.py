#!/usr/bin/env python3
"""
Generate calibration figures for the blocking-justification appendix.

Reads calibration_results_{mean,median}.npz and aggregated_results.csv,
produces:
  - QQ plots (pooled, blocked vs unblocked, one panel per agg)
  - P-value histograms (per dataset, blocked vs unblocked)
  - Eta-squared bar chart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
INSIGHTS_DIR = EXPERIMENT_DIR / "insights"
DEFAULT_CSV = EXPERIMENT_DIR / "aggregated_results" / "aggregated_results.csv"
AGG_NAMES = ["mean", "median"]

ALPHA = 0.05
COLORS = {"blocked": "#4C72B0", "unblocked": "#DD8452"}


def load_calibration(insights_dir: Path) -> dict:
    cal = {}
    for agg in AGG_NAMES:
        path = insights_dir / f"calibration_results_{agg}.npz"
        if not path.exists():
            print(f"Missing {path}; skipping {agg}", file=sys.stderr)
            continue
        cal[agg] = np.load(path, allow_pickle=True)
    return cal


def get_usable_datasets(cal_data: dict, df: pd.DataFrame) -> list[str]:
    """Return datasets present in all npz files with complete alt/null blocks."""
    perturbations = sorted(df["perturbation"].unique())
    npz_datasets = None
    for agg in AGG_NAMES:
        if agg not in cal_data:
            continue
        ds = list(cal_data[agg]["datasets"])
        npz_datasets = set(ds) if npz_datasets is None else npz_datasets & set(ds)

    if npz_datasets is None:
        return []

    usable = []
    for name in sorted(npz_datasets):
        sub = df[df["dataset"] == name]
        has_both = all(
            sub.loc[sub["perturbation"] == p, "distribution"].nunique() == 2
            for p in perturbations
            if p in sub["perturbation"].values
        )
        if has_both:
            usable.append(name)
    return usable


def plot_qq(cal_data: dict, datasets: list[str], out_dir: Path):
    """QQ plots: one panel per agg function, blocked vs unblocked overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    for ax, agg in zip(axes, AGG_NAMES):
        cal = cal_data[agg]
        pooled_b = np.concatenate([cal[f"{d}_blocked"] for d in datasets])
        pooled_u = np.concatenate([cal[f"{d}_unblocked"] for d in datasets])

        n = len(pooled_b)
        theoretical = np.linspace(0, 1, n + 2)[1:-1]

        for label, pv, color in [
            ("blocked", pooled_b, COLORS["blocked"]),
            ("unblocked", pooled_u, COLORS["unblocked"]),
        ]:
            observed = np.sort(pv)
            ax.scatter(theoretical, observed, s=2, alpha=0.35, color=color, label=label)

        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Uniform quantiles")
        ax.set_ylabel("Observed p-value quantiles")
        ax.set_title(f"agg = {agg}")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        f"QQ vs Uniform (pooled, {len(datasets)} datasets, "
        f"R={int(cal_data[AGG_NAMES[0]]['R'])}, B={int(cal_data[AGG_NAMES[0]]['B'])})",
        fontsize=11,
    )
    plt.tight_layout()
    path = out_dir / "qq_calibration.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_pvalue_histograms(cal_data: dict, datasets: list[str], out_dir: Path):
    """P-value histograms per dataset, blocked vs unblocked, one figure per agg."""
    for agg in AGG_NAMES:
        if agg not in cal_data:
            continue
        cal = cal_data[agg]
        n_ds = len(datasets)
        fig, axes = plt.subplots(n_ds, 2, figsize=(8, 2.5 * n_ds))
        if n_ds == 1:
            axes = axes[np.newaxis, :]

        for i, name in enumerate(datasets):
            for j, method in enumerate(["blocked", "unblocked"]):
                ax = axes[i, j]
                pv = cal[f"{name}_{method}"]
                ax.hist(
                    pv, bins=20, range=(0, 1),
                    color=COLORS[method], edgecolor="white", density=True,
                )
                ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
                ax.set_title(f"{name} - {method}", fontsize=9)
                if j == 0:
                    ax.set_ylabel("density")
                if i == n_ds - 1:
                    ax.set_xlabel("p-value")
                rej = np.mean(pv <= ALPHA)
                ax.text(
                    0.95, 0.92, f"rej={rej:.3f}",
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="top",
                )

        R = int(cal["R"])
        B = int(cal["B"])
        fig.suptitle(
            f"Null p-value distributions, agg={agg} (R={R}, B={B})",
            y=1.01, fontsize=11,
        )
        plt.tight_layout()
        path = out_dir / f"pvalue_hist_{agg}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def plot_eta_squared(df: pd.DataFrame, out_dir: Path):
    """Eta-squared bar chart for perturbation type."""
    df_filt = df[df["distribution"].isin(["alt", "null"])].copy()
    datasets = sorted(df_filt["dataset"].unique())

    rows = []
    ss_pert_total, ss_t_total = 0.0, 0.0
    for name in datasets:
        sub = df_filt[df_filt["dataset"] == name]
        y = sub["response"].values.astype(float)
        grand_mean = y.mean()
        ss_t = np.sum((y - grand_mean) ** 2)

        ss_pert = 0.0
        for _, block in sub.groupby("perturbation"):
            block_y = block["response"].values.astype(float)
            ss_pert += len(block_y) * (block_y.mean() - grand_mean) ** 2

        eta2 = ss_pert / ss_t if ss_t > 0 else 0.0
        rows.append({"dataset": name, "eta_sq": eta2})
        ss_pert_total += ss_pert
        ss_t_total += ss_t

    pooled_eta2 = ss_pert_total / ss_t_total
    eta_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(eta_df["dataset"], eta_df["eta_sq"], color="#4C72B0", edgecolor="white")
    ax.axhline(pooled_eta2, color="#DD8452", linestyle="--", label=f"pooled ({pooled_eta2:.3f})")
    ax.set_ylabel("eta-squared")
    ax.set_title("Variance in scores explained by perturbation type")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    path = out_dir / "eta_squared.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # also print the table
    rows.append({"dataset": "POOLED", "eta_sq": pooled_eta2})
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate calibration appendix figures.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--out-dir", type=Path, default=INSIGHTS_DIR / "figures",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, keep_default_na=False)
    cal_data = load_calibration(INSIGHTS_DIR)

    if not cal_data:
        print("No calibration npz files found.", file=sys.stderr)
        return 1

    datasets = get_usable_datasets(cal_data, df)
    print(f"Usable datasets: {datasets}")

    if not datasets:
        print("No usable datasets.", file=sys.stderr)
        return 1

    # Print rejection rate summary
    for agg in AGG_NAMES:
        if agg not in cal_data:
            continue
        cal = cal_data[agg]
        print(f"\n  agg = {agg}")
        for name in datasets:
            pb = cal[f"{name}_blocked"]
            pu = cal[f"{name}_unblocked"]
            print(f"    {name}: blocked={np.mean(pb <= ALPHA):.3f}  unblocked={np.mean(pu <= ALPHA):.3f}")

    plot_qq(cal_data, datasets, args.out_dir)
    plot_pvalue_histograms(cal_data, datasets, args.out_dir)
    plot_eta_squared(df, args.out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
