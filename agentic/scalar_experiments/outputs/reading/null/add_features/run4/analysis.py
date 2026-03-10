import json
import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = "reading.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Filter dyslexia (binary) if available, else dyslexia>0
if "dyslexia_bin" in _df.columns:
    dys_df = _df[_df["dyslexia_bin"] == 1].copy()
else:
    dys_df = _df[_df["dyslexia"] > 0].copy()

# Ensure reader_view and speed
for col in ["reader_view", "speed", "uuid"]:
    if col not in dys_df.columns:
        raise ValueError(f"Missing required column: {col}")

# Drop missing or nonpositive speed
analysis_df = dys_df["uuid reader_view speed".split()].dropna()
analysis_df = analysis_df[analysis_df["speed"] > 0].copy()

# Overall group comparison (not accounting for repeated measures)
rv1 = analysis_df[analysis_df["reader_view"] == 1]["speed"].values
rv0 = analysis_df[analysis_df["reader_view"] == 0]["speed"].values

# Log-transform to mitigate skew
log_rv1 = np.log(rv1)
log_rv0 = np.log(rv0)

# Welch t-test on log speed
welch_t = stats.ttest_ind(log_rv1, log_rv0, equal_var=False)

# Mann-Whitney U test on raw speed
mw = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")

# Effect size (Cohen's d) on log speed
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled)

log_d = cohens_d(log_rv1, log_rv0) if len(log_rv1) > 1 and len(log_rv0) > 1 else np.nan

# Compute descriptive stats
summary = {
    "n_rows": len(analysis_df),
    "n_rv1": len(rv1),
    "n_rv0": len(rv0),
    "mean_speed_rv1": float(np.mean(rv1)) if len(rv1) else np.nan,
    "mean_speed_rv0": float(np.mean(rv0)) if len(rv0) else np.nan,
    "median_speed_rv1": float(np.median(rv1)) if len(rv1) else np.nan,
    "median_speed_rv0": float(np.median(rv0)) if len(rv0) else np.nan,
    "welch_t_stat": float(welch_t.statistic),
    "welch_t_p": float(welch_t.pvalue),
    "mw_u_stat": float(mw.statistic),
    "mw_u_p": float(mw.pvalue),
    "log_cohens_d": float(log_d),
}

# Participant-level paired analysis
# Aggregate per participant and reader_view
agg = (
    analysis_df
    .groupby(["uuid", "reader_view"], as_index=False)
    .agg(mean_speed=("speed", "mean"), median_speed=("speed", "median"), n_obs=("speed", "size"))
)

# Keep participants with both conditions
counts = agg.groupby("uuid")["reader_view"].nunique()
paired_uuids = counts[counts == 2].index
paired = agg[agg["uuid"].isin(paired_uuids)].copy()

if len(paired_uuids) > 1:
    # Pivot to wide
    wide = paired.pivot(index="uuid", columns="reader_view", values="mean_speed")
    wide = wide.dropna()
    # Paired t-test on log speed means
    log_1 = np.log(wide[1])
    log_0 = np.log(wide[0])
    paired_t = stats.ttest_rel(log_1, log_0)
    # Wilcoxon signed-rank
    try:
        wil = stats.wilcoxon(log_1, log_0, alternative="two-sided")
    except ValueError:
        wil = None
    paired_summary = {
        "n_participants_both": int(wide.shape[0]),
        "mean_speed_rv1": float(wide[1].mean()),
        "mean_speed_rv0": float(wide[0].mean()),
        "median_speed_rv1": float(wide[1].median()),
        "median_speed_rv0": float(wide[0].median()),
        "paired_t_stat": float(paired_t.statistic),
        "paired_t_p": float(paired_t.pvalue),
        "wilcoxon_stat": float(wil.statistic) if wil is not None else np.nan,
        "wilcoxon_p": float(wil.pvalue) if wil is not None else np.nan,
    }
else:
    paired_summary = {
        "n_participants_both": int(len(paired_uuids)),
        "mean_speed_rv1": np.nan,
        "mean_speed_rv0": np.nan,
        "median_speed_rv1": np.nan,
        "median_speed_rv0": np.nan,
        "paired_t_stat": np.nan,
        "paired_t_p": np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_p": np.nan,
    }

out = {
    "summary": summary,
    "paired_summary": paired_summary,
}

with open("analysis_results.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
