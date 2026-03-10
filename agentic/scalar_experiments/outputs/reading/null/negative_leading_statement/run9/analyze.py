import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Define dyslexia subset
# Use dyslexia_bin if present; include dyslexia levels 1 or 2
if "dyslexia_bin" in df.columns:
    dys_df = df[df["dyslexia_bin"] == 1].copy()
else:
    dys_df = df[df["dyslexia"].isin([1, 2])].copy()

# Basic cleanup: drop missing speed or reader_view
subset = dys_df.dropna(subset=["speed", "reader_view"]).copy()

# Remove non-positive speeds for log transform
subset = subset[subset["speed"] > 0].copy()

# Group stats
stats_by_rv = subset.groupby("reader_view").agg(
    n=("speed", "count"),
    mean_speed=("speed", "mean"),
    median_speed=("speed", "median"),
    std_speed=("speed", "std"),
).reset_index()

# Welch t-test on log(speed)
subset["log_speed"] = np.log(subset["speed"])
rv0 = subset[subset["reader_view"] == 0]["log_speed"]
rv1 = subset[subset["reader_view"] == 1]["log_speed"]

t_stat, t_p = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U on speed
u_stat, u_p = stats.mannwhitneyu(
    subset[subset["reader_view"] == 1]["speed"],
    subset[subset["reader_view"] == 0]["speed"],
    alternative='two-sided'
)

# Effect size: log-speed mean difference and Cohen's d (on log speed)
mean_diff_log = rv1.mean() - rv0.mean()
# pooled SD for Cohen's d
n1, n0 = rv1.shape[0], rv0.shape[0]
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
cohens_d = mean_diff_log / pooled_sd if pooled_sd > 0 else np.nan

# Regression with covariates (log speed)
# Use only columns that exist and have enough variation
formula_terms = ["reader_view"]

# Potential covariates
categoricals = ["page_id", "device", "education", "gender", "language", "english_native"]
continuous = ["num_words", "Flesch_Kincaid", "age", "correct_rate", "scrolling_time", "running_time", "adjusted_running_time", "img_width", "retake_trial"]

for col in categoricals:
    if col in subset.columns and subset[col].nunique(dropna=True) > 1:
        formula_terms.append(f"C({col})")

for col in continuous:
    if col in subset.columns and subset[col].nunique(dropna=True) > 1:
        formula_terms.append(col)

# Build formula
formula = "log_speed ~ " + " + ".join(formula_terms)

# Drop rows with missing values in model columns
model_cols = ["log_speed", "reader_view"]
for col in categoricals + continuous:
    if col in subset.columns:
        model_cols.append(col)

model_df = subset[model_cols].dropna().copy()

reg_result = None
if model_df.shape[0] >= 20:
    try:
        reg_result = smf.ols(formula, data=model_df).fit(cov_type="HC3")
    except Exception:
        reg_result = None

results = {
    "n_total_dyslexia": int(subset.shape[0]),
    "n_reader_view_0": int((subset["reader_view"] == 0).sum()),
    "n_reader_view_1": int((subset["reader_view"] == 1).sum()),
    "group_stats": stats_by_rv.to_dict(orient="records"),
    "ttest_log_speed": {"t_stat": float(t_stat), "p_value": float(t_p)},
    "mannwhitney_speed": {"u_stat": float(u_stat), "p_value": float(u_p)},
    "mean_diff_log_speed": float(mean_diff_log),
    "cohens_d_log_speed": float(cohens_d),
}

if reg_result is not None:
    coef = reg_result.params.get("reader_view", np.nan)
    pval = reg_result.pvalues.get("reader_view", np.nan)
    results["regression"] = {
        "n_model": int(reg_result.nobs),
        "coef_reader_view": float(coef),
        "p_value_reader_view": float(pval),
        "r2": float(reg_result.rsquared),
    }

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
