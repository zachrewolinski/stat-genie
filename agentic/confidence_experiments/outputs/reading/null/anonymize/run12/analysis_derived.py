import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Rename columns to meaningful names
col_map = {
    "feature1": "participant_id",
    "feature2": "page_id",
    "feature3": "reader_view",
    "feature4": "time_total_ms",
    "feature5": "time_reading_ms",
    "feature6": "time_scrolling_ms",
    "feature7": "word_count",
    "feature17": "dyslexia_binary",
}

df = df.rename(columns=col_map)

# Derived reading speeds (words per minute)
# Use time on page (total) and time excluding scrolling as two sensitivity checks
# Avoid division by zero
speed_total = df["word_count"] * 60000.0 / df["time_total_ms"].replace(0, np.nan)
speed_reading = df["word_count"] * 60000.0 / df["time_reading_ms"].replace(0, np.nan)

# Subset to dyslexia participants
sub = df[df["dyslexia_binary"] == 1].copy()
sub = sub[sub["reader_view"].isin([0, 1])]

# Attach speeds
sub = sub.assign(speed_total=speed_total.loc[sub.index], speed_reading=speed_reading.loc[sub.index])

# Helper for stats

def compare_speed(speed_col):
    s = sub[["participant_id", "reader_view", speed_col]].dropna()
    rv0 = s[s["reader_view"] == 0][speed_col]
    rv1 = s[s["reader_view"] == 1][speed_col]

    n0 = rv0.shape[0]
    n1 = rv1.shape[0]
    mean0 = rv0.mean()
    mean1 = rv1.mean()
    diff = mean1 - mean0

    # Welch t-test
    if n0 > 1 and n1 > 1:
        t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
    else:
        t_stat, p_val = np.nan, np.nan

    # Hedges g
    s0 = rv0.std(ddof=1)
    s1 = rv1.std(ddof=1)
    if n0 > 1 and n1 > 1 and np.isfinite(s0) and np.isfinite(s1):
        pooled = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
        d = (mean1 - mean0) / pooled if pooled > 0 else np.nan
        dfree = n0 + n1 - 2
        J = 1 - (3 / (4 * dfree - 1)) if dfree > 1 else 1
        g = d * J
    else:
        g = np.nan

    # Mixed effects model
    mixed_beta = np.nan
    mixed_p = np.nan
    try:
        if s["participant_id"].nunique() > 1 and s["reader_view"].nunique() == 2:
            md = smf.mixedlm(f"{speed_col} ~ reader_view", s, groups=s["participant_id"])
            mdf = md.fit(reml=False, method="lbfgs")
            mixed_beta = mdf.params.get("reader_view", np.nan)
            mixed_p = mdf.pvalues.get("reader_view", np.nan)
    except Exception:
        pass

    return {
        "n_total": int(s.shape[0]),
        "n_rv0": int(n0),
        "n_rv1": int(n1),
        "mean_rv0": float(mean0),
        "mean_rv1": float(mean1),
        "mean_diff": float(diff),
        "t_stat": float(t_stat) if np.isfinite(t_stat) else None,
        "p_val": float(p_val) if np.isfinite(p_val) else None,
        "hedges_g": float(g) if np.isfinite(g) else None,
        "mixed_beta": float(mixed_beta) if np.isfinite(mixed_beta) else None,
        "mixed_p": float(mixed_p) if np.isfinite(mixed_p) else None,
    }

results = {
    "speed_total_wpm": compare_speed("speed_total"),
    "speed_reading_wpm": compare_speed("speed_reading"),
}

with open("analysis_derived_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
