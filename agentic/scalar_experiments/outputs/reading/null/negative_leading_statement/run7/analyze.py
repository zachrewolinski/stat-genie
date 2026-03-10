import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns
for col in ["speed", "reader_view", "dyslexia_bin", "dyslexia", "num_words", "adjusted_running_time", "running_time", "scrolling_time", "correct_rate", "age"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Focus on dyslexic participants (dyslexia_bin == 1)
df_dys = df[df["dyslexia_bin"] == 1].copy()

# Drop missing values for speed and reader_view
subset = df_dys.dropna(subset=["speed", "reader_view", "uuid", "page_id", "num_words", "language", "device", "age"])

# Summary stats
summary = subset.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).to_dict()

# Welch's t-test
rv1 = subset[subset["reader_view"] == 1]["speed"].dropna()
rv0 = subset[subset["reader_view"] == 0]["speed"].dropna()

# Handle small samples
if len(rv1) > 1 and len(rv0) > 1:
    t_res = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
else:
    t_res = None

# Mann-Whitney U (nonparametric)
if len(rv1) > 0 and len(rv0) > 0:
    try:
        u_res = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")
    except Exception:
        u_res = None
else:
    u_res = None

# Effect size: Cohen's d (using pooled SD for independent samples)
if len(rv1) > 1 and len(rv0) > 1:
    n1, n0 = len(rv1), len(rv0)
    s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    d = (rv1.mean() - rv0.mean()) / pooled if pooled > 0 else np.nan
else:
    d = np.nan

# Also consider log-speed due to heavy skew
subset = subset.copy()
subset["log_speed"] = np.log(subset["speed"].clip(lower=1e-6))

rv1_log = subset[subset["reader_view"] == 1]["log_speed"].dropna()
rv0_log = subset[subset["reader_view"] == 0]["log_speed"].dropna()
if len(rv1_log) > 1 and len(rv0_log) > 1:
    t_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy="omit")
else:
    t_log = None

# Regression with controls (simple OLS with clustered SE by uuid)
# Model: log_speed ~ reader_view + num_words + C(page_id) + C(language) + C(device) + age
# Use only rows with needed columns
model_data = subset.dropna(subset=["log_speed", "reader_view", "num_words", "page_id", "language", "device", "age", "uuid"]).copy()
model_result = None
if len(model_data) > 50:  # ensure enough data
    try:
        model = smf.ols("log_speed ~ reader_view + num_words + C(page_id) + C(language) + C(device) + age", data=model_data)
        model_result = model.fit(cov_type="cluster", cov_kwds={"groups": model_data["uuid"]})
    except Exception:
        model_result = None

# Simple within-subject comparison: for each uuid, compare mean speed under reader_view 1 vs 0
# Only keep participants with both conditions
pivot = subset.pivot_table(index="uuid", columns="reader_view", values="log_speed", aggfunc="mean")
within = pivot.dropna()
within_t = None
if within.shape[0] > 1:
    within_t = stats.ttest_rel(within[1], within[0])

results = {
    "n_total": int(len(df)),
    "n_dyslexia": int(len(df_dys)),
    "summary_speed_by_reader_view": summary,
    "t_test": None if t_res is None else {"statistic": float(t_res.statistic), "pvalue": float(t_res.pvalue)},
    "mannwhitney": None if u_res is None else {"statistic": float(u_res.statistic), "pvalue": float(u_res.pvalue)},
    "cohens_d": None if np.isnan(d) else float(d),
    "t_test_log": None if t_log is None else {"statistic": float(t_log.statistic), "pvalue": float(t_log.pvalue)},
    "regression": None,
    "within_subject_t": None if within_t is None else {"statistic": float(within_t.statistic), "pvalue": float(within_t.pvalue), "n": int(within.shape[0])},
}
if model_result is not None:
    coef = model_result.params.get("reader_view", np.nan)
    pval = model_result.pvalues.get("reader_view", np.nan)
    results["regression"] = {"coef": float(coef), "pvalue": float(pval), "n": int(model_data.shape[0])}

print(json.dumps(results, indent=2))
