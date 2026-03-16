import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Determine dyslexia indicator
if "dyslexia_bin" in df.columns:
    dys_mask = df["dyslexia_bin"] == 1
elif "dyslexia" in df.columns:
    dys_mask = df["dyslexia"] > 0
else:
    raise ValueError("No dyslexia indicator found.")

df_dys = df.loc[dys_mask].copy()

# Keep positive speed
if "speed" not in df_dys.columns:
    raise ValueError("speed column missing")

df_dys = df_dys[df_dys["speed"].notna() & (df_dys["speed"] > 0)]

# Group summaries
rv1 = df_dys[df_dys["reader_view"] == 1]["speed"].astype(float)
rv0 = df_dys[df_dys["reader_view"] == 0]["speed"].astype(float)

n1 = rv1.shape[0]
n0 = rv0.shape[0]
mean1 = rv1.mean()
mean0 = rv0.mean()
median1 = rv1.median()
median0 = rv0.median()
std1 = rv1.std(ddof=1)
std0 = rv0.std(ddof=1)

# Effect size (Cohen's d)
if n1 > 1 and n0 > 1:
    s_pooled = np.sqrt(((n1 - 1) * std1 ** 2 + (n0 - 1) * std0 ** 2) / (n1 + n0 - 2))
    cohens_d = (mean1 - mean0) / s_pooled if s_pooled > 0 else np.nan
else:
    cohens_d = np.nan

# Welch t-test
try:
    t_stat, t_p = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
except Exception:
    t_stat, t_p = np.nan, np.nan

# Mann-Whitney U
try:
    u_stat, u_p = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")
except Exception:
    u_stat, u_p = np.nan, np.nan

# Log-speed analysis
log_rv1 = np.log(rv1)
log_rv0 = np.log(rv0)
try:
    t_stat_log, t_p_log = stats.ttest_ind(log_rv1, log_rv0, equal_var=False, nan_policy="omit")
except Exception:
    t_stat_log, t_p_log = np.nan, np.nan

# Regression with clustered SEs by participant
# Prepare model data
model_cols = [
    "speed",
    "reader_view",
    "num_words",
    "Flesch_Kincaid",
    "page_id",
    "device",
    "age",
    "gender",
    "education",
    "language",
    "english_native",
    "retake_trial",
    "uuid",
]
if "dyslexia" in df_dys.columns:
    model_cols.append("dyslexia")

model_df = df_dys[model_cols].copy()
model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
model_df["log_speed"] = np.log(model_df["speed"].astype(float))

# Build formula
formula_parts = [
    "reader_view",
    "num_words",
    "Flesch_Kincaid",
    "C(page_id)",
    "C(device)",
    "age",
    "C(gender)",
    "C(education)",
    "C(language)",
    "C(english_native)",
    "retake_trial",
]
if "dyslexia" in model_df.columns:
    formula_parts.append("dyslexia")

formula = "log_speed ~ " + " + ".join(formula_parts)

reg_result = None
reg_summary = {}
if model_df.shape[0] > 0:
    model = smf.ols(formula=formula, data=model_df)
    reg_result = model.fit(cov_type="cluster", cov_kwds={"groups": model_df["uuid"]})
    coef = reg_result.params.get("reader_view", np.nan)
    se = reg_result.bse.get("reader_view", np.nan)
    pval = reg_result.pvalues.get("reader_view", np.nan)
    # 95% CI
    if np.isfinite(coef) and np.isfinite(se):
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
    else:
        ci_low = np.nan
        ci_high = np.nan

    reg_summary = {
        "coef": float(coef),
        "se": float(se),
        "pval": float(pval),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": int(model_df.shape[0]),
    }

# Output summary for inspection
summary = {
    "n_total": int(df_dys.shape[0]),
    "n_reader_view_on": int(n1),
    "n_reader_view_off": int(n0),
    "mean_speed_on": float(mean1),
    "mean_speed_off": float(mean0),
    "median_speed_on": float(median1),
    "median_speed_off": float(median0),
    "std_speed_on": float(std1),
    "std_speed_off": float(std0),
    "cohens_d": float(cohens_d),
    "t_p": float(t_p),
    "u_p": float(u_p),
    "t_p_log": float(t_p_log),
    "regression": reg_summary,
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Also save a short text summary
with open("analysis_summary.txt", "w") as f:
    f.write(json.dumps(summary, indent=2))
