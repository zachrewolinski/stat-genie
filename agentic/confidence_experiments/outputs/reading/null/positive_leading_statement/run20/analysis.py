import json
import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = "reading.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Basic cleaning: keep rows with speed and reader_view
_df = _df.copy()

# Define dyslexia group: dyslexia_bin == 1 OR dyslexia in {1,2}
# Prefer dyslexia_bin when present; fall back to dyslexia if needed
if "dyslexia_bin" in _df.columns:
    dys_mask = _df["dyslexia_bin"].astype(float) == 1.0
else:
    dys_mask = _df["dyslexia"].astype(float).isin([1.0, 2.0])

# Ensure reader_view and speed are numeric
_df["reader_view"] = pd.to_numeric(_df["reader_view"], errors="coerce")
_df["speed"] = pd.to_numeric(_df["speed"], errors="coerce")

# Keep dyslexia rows with valid speed and reader_view
_df_dys = _df.loc[dys_mask & _df["reader_view"].isin([0, 1]) & _df["speed"].notna()].copy()

# Optional: drop extreme speed outliers (top/bottom 0.5%) for robustness check
# We'll compute stats both with and without trimming

def trim_outliers(df, col, lower_q=0.005, upper_q=0.995):
    lo = df[col].quantile(lower_q)
    hi = df[col].quantile(upper_q)
    return df[(df[col] >= lo) & (df[col] <= hi)].copy()


def summarize_group(df, label):
    g0 = df[df["reader_view"] == 0]["speed"]
    g1 = df[df["reader_view"] == 1]["speed"]
    summary = {
        "label": label,
        "n_no": int(g0.shape[0]),
        "n_yes": int(g1.shape[0]),
        "mean_no": float(g0.mean()),
        "mean_yes": float(g1.mean()),
        "median_no": float(g0.median()),
        "median_yes": float(g1.median()),
    }

    # Log-transform for skew
    g0_log = np.log(g0)
    g1_log = np.log(g1)

    # Welch t-test on log speed
    t_stat, p_val = stats.ttest_ind(g1_log, g0_log, equal_var=False, nan_policy="omit")

    # Mann-Whitney U on raw speed
    try:
        u_stat, p_u = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    except ValueError:
        u_stat, p_u = np.nan, np.nan

    # Effect size: Cohen's d on log speed
    def cohens_d(x, y):
        nx = len(x)
        ny = len(y)
        vx = np.var(x, ddof=1)
        vy = np.var(y, ddof=1)
        pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
        return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)

    d = cohens_d(g1_log, g0_log)

    # Ratio of geometric means (exp of mean difference in logs)
    ratio = float(np.exp(np.mean(g1_log) - np.mean(g0_log)))

    summary.update({
        "t_stat_log": float(t_stat),
        "p_val_log": float(p_val),
        "u_stat": float(u_stat) if not np.isnan(u_stat) else None,
        "p_val_u": float(p_u) if not np.isnan(p_u) else None,
        "cohens_d_log": float(d),
        "geo_mean_ratio": ratio,
    })

    return summary

# Main summary
summary_all = summarize_group(_df_dys, "all_dyslexia")

# Exclude retake_trial if present
summary_no_retake = None
if "retake_trial" in _df_dys.columns:
    df_no_retake = _df_dys[_df_dys["retake_trial"] != 1].copy()
    summary_no_retake = summarize_group(df_no_retake, "no_retake")

# Trim outliers robustness
summary_trim = summarize_group(trim_outliers(_df_dys, "speed"), "trimmed_0.5pct")

out = {
    "summary_all": summary_all,
    "summary_no_retake": summary_no_retake,
    "summary_trim": summary_trim,
}

print(json.dumps(out, indent=2))
