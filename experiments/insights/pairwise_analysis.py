import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


SCORE_COLS = [
    "Independent Variables Similarity Score",
    "Control Variables Similarity Score",
    "Response Variables Similarity Score",
    "Model Similarity Score",
    "conclusions",
    "overall_similarity",
]
FACTOR_COLS = [c for c in SCORE_COLS if c != "overall_similarity"]

PERTURB_BY_INDEX = {
    0: "add_features_output",
    1: "anonymize_output",
    2: "noperturb_output",
    3: "replace_with_rvs_output",
    4: "shuffle_names_output",
}

def parse_pair_key(k: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse inner key like '0_3' -> (0,3)."""
    try:
        a, b = k.split("_")
        return int(a), int(b)
    except Exception:
        return None, None


def parse_outer_key(k: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse outer key like '0_1' -> (0,1)."""
    try:
        a, b = k.split("_")
        return int(a), int(b)
    except Exception:
        return None, None


def mean_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(x))
    if x.size == 1:
        return m, np.nan, np.nan
    se = float(np.std(x, ddof=1) / np.sqrt(x.size))
    tcrit = stats.t.ppf(1 - alpha / 2, df=x.size - 1)
    return m, m - tcrit * se, m + tcrit * se


def ensure_out_folder(out_folder: str) -> Path:
    p = Path(out_folder)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_pearsonr(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    """
    Pearson r with guards:
    - returns (nan, nan, n) if fewer than 3 points or if x or y is constant
    """
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)
    if np.nanstd(xv) == 0.0 or np.nanstd(yv) == 0.0:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(xv, yv)
    return float(r), float(p), n


def r2_from_lstsq(y: np.ndarray, X: np.ndarray) -> float:
    """
    Compute R^2 for y ~ [1, X] via least squares.
    X should be shape (n, k) without intercept.
    """
    y = y.astype(float)
    X = X.astype(float)
    n = y.shape[0]
    if n < 2:
        return np.nan

    # Add intercept
    X1 = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        # y constant -> "perfect" in a degenerate sense
        return 1.0
    return 1.0 - ss_res / ss_tot


# -------------------------
# Parsing / Flattening
# -------------------------
def flatten_to_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Flatten nested JSON into a dataframe with:
    - perturb_i, perturb_j (outer perturbation names)
    - run_i, run_j (inner run indices)
    - is_diagonal: run_i == run_j
    """
    rows: List[Dict[str, Any]] = []
    for outer_k, inner_map in data.items():
        if not isinstance(inner_map, dict):
            continue

        oi, oj = parse_outer_key(outer_k)
        if oi is None or oj is None:
            continue
        if oi not in PERTURB_BY_INDEX or oj not in PERTURB_BY_INDEX:
            continue

        perturb_i = PERTURB_BY_INDEX[oi]
        perturb_j = PERTURB_BY_INDEX[oj]

        for inner_k, payload in inner_map.items():
            if not isinstance(payload, dict):
                continue
            i, j = parse_pair_key(inner_k)

            row = {
                "outer_key": outer_k,
                "outer_i": oi,
                "outer_j": oj,
                "perturb_i": perturb_i,
                "perturb_j": perturb_j,
                "perturb_pair": f"{perturb_i}__vs__{perturb_j}",
                "inner_key": inner_k,
                "run_i": i,
                "run_j": j,
                "is_diagonal": (i is not None and j is not None and i == j),
            }
            for c in SCORE_COLS:
                row[c] = payload.get(c, np.nan)

            rows.append(row)

    df = pd.DataFrame(rows)
    for c in SCORE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def within_perturbation_view(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict to within-perturbation blocks (outer_i == outer_j),
    and expose a single 'perturbation' column for grouping.
    """
    w = df_all[df_all["outer_i"] == df_all["outer_j"]].copy()
    w["perturbation"] = w["perturb_i"]
    return w


def per_perturbation_summary(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for perturb, g in df.groupby("perturbation"):
        for label, mask in [
            ("off_diagonal", ~g["is_diagonal"]),
            ("diagonal", g["is_diagonal"]),
        ]:
            x = g.loc[mask, "overall_similarity"].to_numpy()
            m, lo, hi = mean_ci(x)
            n = int(np.sum(~np.isnan(x)))
            out_rows.append(
                {
                    "perturbation": perturb,
                    "subset": label,
                    "mean": m,
                    "ci_low": lo,
                    "ci_high": hi,
                    "std": float(np.nanstd(x, ddof=1)) if n > 1 else np.nan,
                }
            )
    return pd.DataFrame(out_rows).sort_values(
        ["subset", "mean"], ascending=[True, False]
    )


def run_stability_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each perturbation and each run r, compute the mean similarity between
    run r and all other runs (off-diagonal only).
    """
    d = df[~df["is_diagonal"]].copy()
    rows = []
    for perturb, g in d.groupby("perturbation"):
        runs = sorted(
            set(
                g["run_i"].dropna().astype(int).tolist()
                + g["run_j"].dropna().astype(int).tolist()
            )
        )
        for r in runs:
            vals = g[(g["run_i"] == r) | (g["run_j"] == r)][
                "overall_similarity"
            ].dropna()
            rows.append(
                {
                    "perturbation": perturb,
                    "run": r,
                    "n_pairs": int(vals.shape[0]),
                    "mean_similarity_to_others": float(vals.mean())
                    if vals.shape[0]
                    else np.nan,
                    "std": float(vals.std(ddof=1)) if vals.shape[0] > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["perturbation", "mean_similarity_to_others"]
    )


def run_stability_summary(stab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for perturb, g in stab.groupby("perturbation"):
        x = g["mean_similarity_to_others"].dropna().to_numpy()
        m, lo, hi = mean_ci(x)
        rows.append(
            {
                "perturbation": perturb,
                "mean_of_run_means": m,
                "ci_low": lo,
                "ci_high": hi,
                "min_run_mean": float(np.min(x)) if x.size else np.nan,
                "max_run_mean": float(np.max(x)) if x.size else np.nan,
                "std_across_runs": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_of_run_means", ascending=False)


def worst_pairs_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each perturbation, keep only the (possibly multiple) pairs that achieve
    the minimum overall_similarity (off-diagonal only).
    """
    d = df[~df["is_diagonal"]].dropna(subset=["overall_similarity"]).copy()
    out_cols = (
        ["perturbation", "inner_key", "run_i", "run_j", "overall_similarity"]
        + [c for c in SCORE_COLS if c != "overall_similarity"]
    )
    rows = []
    for perturb, g in d.groupby("perturbation"):
        if g.empty:
            continue
        min_val = g["overall_similarity"].min()
        worst = (
            g[g["overall_similarity"] == min_val]
            .copy()
            .sort_values(["overall_similarity", "inner_key"])
        )
        rows.append(worst[out_cols])
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=out_cols)


def cv_per_perturbation(stab_summary: pd.DataFrame) -> pd.DataFrame:
    out = stab_summary.copy()
    out["cv_across_runs"] = out["std_across_runs"] / out["mean_of_run_means"]
    keep = [
        "perturbation",
        "mean_of_run_means",
        "std_across_runs",
        "cv_across_runs",
        "min_run_mean",
        "max_run_mean",
        "ci_low",
        "ci_high",
    ]
    return out[keep].sort_values("cv_across_runs", ascending=False)


def factor_variance_by_perturbation(df: pd.DataFrame) -> pd.DataFrame:
    d = df[~df["is_diagonal"]].copy()
    rows = []
    for perturb, g in d.groupby("perturbation"):
        for metric in SCORE_COLS:
            x = g[metric].dropna().to_numpy()
            n = int(x.size)
            rows.append(
                {
                    "perturbation": perturb,
                    "metric": metric,
                    "mean": float(np.mean(x)) if n else np.nan,
                    "std": float(np.std(x, ddof=1)) if n > 1 else np.nan,
                    "var": float(np.var(x, ddof=1)) if n > 1 else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["perturbation", "metric"])


def factor_correlations(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df[~df["is_diagonal"]].copy()

    # overall pooled
    rows_overall = []
    y = d["overall_similarity"]
    for f in FACTOR_COLS:
        r, p, n = safe_pearsonr(d[f], y)
        rows_overall.append(
            {"scope": "all", "factor": f, "pearson_r": r, "p": p, "n": n}
        )
    overall_corr = (
        pd.DataFrame(rows_overall)
        .sort_values("pearson_r", ascending=False, na_position="last")
    )

    # per perturbation
    rows_by = []
    for perturb, g in d.groupby("perturbation"):
        y = g["overall_similarity"]
        for f in FACTOR_COLS:
            r, p, n = safe_pearsonr(g[f], y)
            rows_by.append(
                {
                    "perturbation": perturb,
                    "factor": f,
                    "pearson_r": r,
                    "p": p,
                    "n": n,
                }
            )
    by_corr = pd.DataFrame(rows_by).sort_values(
        ["perturbation", "pearson_r"], ascending=[True, False], na_position="last"
    )

    return overall_corr, by_corr


def drop_one_r2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop-one R^2 using least squares:
    - Fit full model: overall_similarity ~ all FACTOR_COLS
    - For each factor f, fit model without f and measure delta R^2
    """
    d = df[~df["is_diagonal"]].copy()
    use_cols = ["overall_similarity"] + FACTOR_COLS
    dd = d[use_cols].dropna()

    if dd.shape[0] < 10:
        return pd.DataFrame(columns=["factor", "r2_full", "r2_drop", "delta_r2", "n"])

    y = dd["overall_similarity"].to_numpy(dtype=float)
    X_full = dd[FACTOR_COLS].to_numpy(dtype=float)

    r2_full = r2_from_lstsq(y, X_full)

    rows = []
    for f in FACTOR_COLS:
        kept = [c for c in FACTOR_COLS if c != f]
        X_drop = dd[kept].to_numpy(dtype=float)
        r2_drop = r2_from_lstsq(y, X_drop)
        rows.append(
            {
                "factor": f,
                "r2_full": float(r2_full),
                "r2_drop": float(r2_drop),
                "delta_r2": float(r2_full - r2_drop),
            }
        )

    return pd.DataFrame(rows).sort_values("delta_r2", ascending=False)


def make_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Heatmaps of overall_similarity for each perturbation (within-perturbation only).
    Uses run_i/run_j indices to build an n_runs x n_runs matrix.
    """
    heat_dir = out_dir / "heatmaps"
    heat_dir.mkdir(parents=True, exist_ok=True)

    for perturb, g in df.groupby("perturbation"):
        runs_i = g["run_i"].dropna().astype(int)
        runs_j = g["run_j"].dropna().astype(int)
        if runs_i.empty or runs_j.empty:
            continue

        n_runs = int(max(runs_i.max(), runs_j.max()) + 1)
        mat = np.full((n_runs, n_runs), np.nan, dtype=float)

        for _, row in g.iterrows():
            i, j = row["run_i"], row["run_j"]
            v = row["overall_similarity"]
            if pd.notna(i) and pd.notna(j) and pd.notna(v):
                mat[int(i), int(j)] = float(v)

        fig, ax = plt.subplots()
        im = ax.imshow(mat, aspect="equal")

        ax.set_title(f"Run-pair similarity heatmap: {perturb}")
        ax.set_xlabel("run_j")
        ax.set_ylabel("run_i")
        ax.set_xticks(range(n_runs))
        ax.set_yticks(range(n_runs))

        for i in range(n_runs):
            for j in range(n_runs):
                if not np.isnan(mat[i, j]):
                    ax.text(
                        j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=8
                    )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(heat_dir / f"heatmap_{perturb}.png", dpi=200)
        plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to nested JSON")
    ap.add_argument("--out_folder", required=True, help="Folder to write all outputs into")
    args = ap.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    df_all = flatten_to_df(data)
    if df_all.empty:
        raise SystemExit(
            "No rows found. Check that outer keys look like 'i_j' with i,j in {0..4}."
        )

    df = within_perturbation_view(df_all)
    if df.empty:
        raise SystemExit(
            "No within-perturbation rows found (outer blocks where i==j). "
            "If your JSON only has cross-perturbation blocks, run-stability analyses are not defined."
        )

    out_dir = ensure_out_folder(args.out_folder)

    print("=== Loaded ===")
    print("All rows:", df_all.shape[0])
    print("Within-perturbation rows:", df.shape[0])
    print("Perturbations (within):", sorted(df["perturbation"].unique().tolist()))

    # Summary overall similarity (diagonal vs off-diagonal)
    summ = per_perturbation_summary(df)
    summ.to_csv(out_dir / "summary_overall_similarity.csv", index=False)

    # Run stability (within each perturbation)
    stab = run_stability_table(df)
    stab.to_csv(out_dir / "run_stability_table.csv", index=False)

    stab_summary = run_stability_summary(stab)
    stab_summary.to_csv(out_dir / "run_stability_summary.csv", index=False)

    # Worst pairs (within each perturbation)
    worst = worst_pairs_only(df)
    worst.to_csv(out_dir / "worst_pairs_only.csv", index=False)

    # CV across runs
    cv = cv_per_perturbation(stab_summary)
    cv.to_csv(out_dir / "cv_across_runs.csv", index=False)

    # Factor variance (off-diagonal only)
    fv = factor_variance_by_perturbation(df)
    fv.to_csv(out_dir / "factor_variance_off_diagonal.csv", index=False)

    # Correlations with overall similarity (off-diagonal only)
    corr_all, corr_by = factor_correlations(df)
    corr_all.to_csv(out_dir / "factor_correlation_with_overall_all.csv", index=False)
    corr_by.to_csv(out_dir / "factor_correlation_with_overall_by_perturbation.csv", index=False)

    # Drop-one delta R^2 (off-diagonal only)
    delta_r2 = drop_one_r2(df)
    delta_r2.to_csv(out_dir / "factor_dropone_delta_r2.csv", index=False)

    # Heatmaps (within each perturbation)
    make_heatmaps(df, out_dir)

    print("\nWrote outputs to:", out_dir)
    print(" - summary_overall_similarity.csv")
    print(" - run_stability_table.csv")
    print(" - run_stability_summary.csv")
    print(" - worst_pairs_only.csv")
    print(" - cv_across_runs.csv")
    print(" - factor_variance_off_diagonal.csv")
    print(" - factor_correlation_with_overall_all.csv")
    print(" - factor_correlation_with_overall_by_perturbation.csv")
    print(" - factor_dropone_delta_r2.csv")
    print(" - heatmaps/heatmap_<perturbation>.png")


if __name__ == "__main__":
    main()
