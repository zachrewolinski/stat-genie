import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def cohens_d(x, y):
    # Hedges' g correction for unequal sample sizes
    x = np.asarray(x)
    y = np.asarray(y)
    nx = x.size
    ny = y.size
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled == 0:
        return np.nan
    d = (x.mean() - y.mean()) / np.sqrt(pooled)
    # Hedges' correction
    correction = 1 - (3 / (4 * (nx + ny) - 9))
    return d * correction


def summarize_group(df, label):
    return {
        "n": int(df.shape[0]),
        "mean": float(df["speed"].mean()),
        "median": float(df["speed"].median()),
        "std": float(df["speed"].std(ddof=1)),
        "label": label,
    }


# Load data
full = pd.read_csv("reading.csv")

# Focus on participants with dyslexia
# Prefer dyslexia_bin when available; fallback to dyslexia >= 1
if "dyslexia_bin" in full.columns:
    dyslexia_df = full[full["dyslexia_bin"] == 1].copy()
else:
    dyslexia_df = full[full["dyslexia"] >= 1].copy()

# Ensure necessary columns
required_cols = ["reader_view", "speed"]
missing = [c for c in required_cols if c not in dyslexia_df.columns]
if missing:
    raise ValueError(f"Missing columns in data: {missing}")

# Drop rows with missing or non-positive speed
analysis_df = dyslexia_df.dropna(subset=["reader_view", "speed"]).copy()
analysis_df = analysis_df[analysis_df["speed"] > 0]

# Split by reader_view
rv_on = analysis_df[analysis_df["reader_view"] == 1]
rv_off = analysis_df[analysis_df["reader_view"] == 0]

summary = {
    "rv_on": summarize_group(rv_on, "reader_view=1"),
    "rv_off": summarize_group(rv_off, "reader_view=0"),
}

# Welch t-test on log(speed) to reduce skew
analysis_df["log_speed"] = np.log(analysis_df["speed"])
rv_on_log = analysis_df[analysis_df["reader_view"] == 1]["log_speed"]
rv_off_log = analysis_df[analysis_df["reader_view"] == 0]["log_speed"]

welch_t = stats.ttest_ind(rv_on_log, rv_off_log, equal_var=False, nan_policy="omit")

# Mann-Whitney U on raw speed (robust to non-normality)
try:
    mw_u = stats.mannwhitneyu(rv_on["speed"], rv_off["speed"], alternative="two-sided")
except ValueError:
    mw_u = None

# Effect size on log-speed (Hedges' g)
hedges_g = cohens_d(rv_on_log, rv_off_log)

# Regression controlling for page_id and device and num_words (if available)
control_vars = []
if "num_words" in analysis_df.columns:
    control_vars.append("num_words")
if "page_id" in analysis_df.columns:
    control_vars.append("C(page_id)")
if "device" in analysis_df.columns:
    control_vars.append("C(device)")
if "age" in analysis_df.columns:
    control_vars.append("age")
if "english_native" in analysis_df.columns:
    control_vars.append("C(english_native)")

formula = "log_speed ~ reader_view"
if control_vars:
    formula += " + " + " + ".join(control_vars)

# Fit only if enough data
regression_result = None
if analysis_df.shape[0] >= 30:
    model = smf.ols(formula, data=analysis_df).fit()
    regression_result = {
        "coef_reader_view": float(model.params.get("reader_view", np.nan)),
        "p_reader_view": float(model.pvalues.get("reader_view", np.nan)),
        "n": int(model.nobs),
        "formula": formula,
        "r2": float(model.rsquared),
    }

results = {
    "summary": summary,
    "welch_t_log_speed": {
        "statistic": float(welch_t.statistic),
        "pvalue": float(welch_t.pvalue),
    },
    "mannwhitney_speed": None if mw_u is None else {
        "statistic": float(mw_u.statistic),
        "pvalue": float(mw_u.pvalue),
    },
    "hedges_g_log_speed": None if np.isnan(hedges_g) else float(hedges_g),
    "regression": regression_result,
}

print(json.dumps(results, indent=2))
