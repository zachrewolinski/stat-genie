import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = "reading.csv"

df = pd.read_csv(path)

# Ensure numeric columns
for col in ["reader_view", "speed", "dyslexia_bin", "dyslexia", "num_words", "Flesch_Kincaid"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Define dyslexia flag
# Prefer dyslexia_bin when available; fall back to dyslexia values 1 or 2 when dyslexia_bin is missing.
if "dyslexia_bin" in df.columns:
    dys_bin = df["dyslexia_bin"]
else:
    dys_bin = pd.Series(np.nan, index=df.index)

if "dyslexia" in df.columns:
    dys = df["dyslexia"]
else:
    dys = pd.Series(np.nan, index=df.index)

# Create dyslexic mask
mask_dys = (dys_bin == 1)
mask_dys = mask_dys | (dys_bin.isna() & dys.isin([1, 2]))

# Keep rows with speed and reader_view
df_dys = df.loc[mask_dys].copy()

df_dys = df_dys.loc[df_dys["reader_view"].isin([0, 1])]

df_dys = df_dys.loc[df_dys["speed"].notna()]

# Basic stats
n_total = len(df_dys)

# Split groups
rv1 = df_dys.loc[df_dys["reader_view"] == 1, "speed"].astype(float)
rv0 = df_dys.loc[df_dys["reader_view"] == 0, "speed"].astype(float)

n1 = len(rv1)
n0 = len(rv0)

# Handle potential zeros or negatives for log; speed should be positive
rv1_log = np.log(rv1)
rv0_log = np.log(rv0)

# Welch t-test on log speed
if n1 > 1 and n0 > 1:
    t_stat, p_val = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy="omit")
else:
    t_stat, p_val = np.nan, np.nan

# Mann-Whitney U test on raw speed
if n1 > 0 and n0 > 0:
    try:
        u_stat, p_u = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")
    except Exception:
        u_stat, p_u = np.nan, np.nan
else:
    u_stat, p_u = np.nan, np.nan

# Effect size (Cohen's d) on log speed
if n1 > 1 and n0 > 1:
    mean1 = rv1_log.mean()
    mean0 = rv0_log.mean()
    s1 = rv1_log.std(ddof=1)
    s0 = rv0_log.std(ddof=1)
    # pooled SD for unequal n
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    if s_pooled > 0:
        d = (mean1 - mean0) / s_pooled
    else:
        d = np.nan
else:
    d = np.nan

# Group summaries
summary = {
    "n_total_dys": int(n_total),
    "n_reader_view_1": int(n1),
    "n_reader_view_0": int(n0),
    "mean_speed_rv1": float(rv1.mean()) if n1 > 0 else np.nan,
    "mean_speed_rv0": float(rv0.mean()) if n0 > 0 else np.nan,
    "median_speed_rv1": float(rv1.median()) if n1 > 0 else np.nan,
    "median_speed_rv0": float(rv0.median()) if n0 > 0 else np.nan,
    "t_stat_log": float(t_stat) if np.isfinite(t_stat) else np.nan,
    "p_val_log": float(p_val) if np.isfinite(p_val) else np.nan,
    "u_stat": float(u_stat) if np.isfinite(u_stat) else np.nan,
    "p_u": float(p_u) if np.isfinite(p_u) else np.nan,
    "cohens_d_log": float(d) if np.isfinite(d) else np.nan,
}

# Regression with controls (log speed) if possible
reg_result = None
try:
    # Keep rows with needed columns
    reg_df = df_dys.copy()
    reg_df = reg_df.loc[reg_df["num_words"].notna() & reg_df["Flesch_Kincaid"].notna()]
    # log speed
    reg_df = reg_df.loc[reg_df["speed"] > 0]
    reg_df["log_speed"] = np.log(reg_df["speed"])

    # Simple model with reader_view + controls + page_id fixed effects
    # Use categorical for page_id and device where available
    formula = "log_speed ~ reader_view + num_words + Flesch_Kincaid"
    if "page_id" in reg_df.columns:
        formula += " + C(page_id)"
    if "device" in reg_df.columns:
        formula += " + C(device)"

    model = smf.ols(formula, data=reg_df).fit(cov_type="HC3")
    reg_result = {
        "n_reg": int(model.nobs),
        "coef_reader_view": float(model.params.get("reader_view", np.nan)),
        "se_reader_view": float(model.bse.get("reader_view", np.nan)),
        "p_reader_view": float(model.pvalues.get("reader_view", np.nan)),
    }
except Exception:
    reg_result = None

output = {"summary": summary, "regression": reg_result}

with open("analysis_output.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
