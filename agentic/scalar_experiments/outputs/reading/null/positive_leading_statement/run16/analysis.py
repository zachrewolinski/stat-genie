import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv("reading.csv")

# Define dyslexia group
if "dyslexia_bin" in df.columns:
    dys = df[df["dyslexia_bin"] == 1].copy()
else:
    dys = df[df["dyslexia"].isin([1, 2])].copy()

# Basic group stats
rv0 = dys[dys["reader_view"] == 0]
rv1 = dys[dys["reader_view"] == 1]

stats_basic = {
    "n_total": int(len(dys)),
    "n_reader_view_0": int(len(rv0)),
    "n_reader_view_1": int(len(rv1)),
    "mean_speed_rv0": float(rv0["speed"].mean()),
    "mean_speed_rv1": float(rv1["speed"].mean()),
    "median_speed_rv0": float(rv0["speed"].median()),
    "median_speed_rv1": float(rv1["speed"].median()),
}

# Welch t-test on raw speed
if len(rv0) > 1 and len(rv1) > 1:
    t_res = stats.ttest_ind(rv1["speed"], rv0["speed"], equal_var=False, nan_policy="omit")
    t_raw = {"t": float(t_res.statistic), "p": float(t_res.pvalue)}
else:
    t_raw = {"t": np.nan, "p": np.nan}

# Effect size (Cohen's d) on raw speed
m1 = rv1["speed"].mean()
m0 = rv0["speed"].mean()
s1 = rv1["speed"].std()
s0 = rv0["speed"].std()

if np.isfinite(s1) and np.isfinite(s0):
    n1 = len(rv1)
    n0 = len(rv0)
    if n1 > 1 and n0 > 1:
        pooled = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
        d = (m1 - m0) / pooled if pooled > 0 else np.nan
    else:
        d = np.nan
else:
    d = np.nan

# Log-speed for regression (avoid log(0))
# speed seems positive; still guard.
min_speed = dys["speed"].min()
if min_speed <= 0:
    dys = dys[dys["speed"] > 0].copy()

# Add log speed
import numpy as _np

dys["log_speed"] = np.log(dys["speed"])

# Regression with cluster-robust SE by uuid to handle repeated measures
formula = (
    "log_speed ~ reader_view + C(page_id) + num_words + C(device) + age + "
    "C(gender) + C(education) + C(english_native) + retake_trial"
)

# Drop rows with missing values in model variables to keep group length aligned
model_vars = [
    "log_speed",
    "reader_view",
    "page_id",
    "num_words",
    "device",
    "age",
    "gender",
    "education",
    "english_native",
    "retake_trial",
    "uuid",
]
model_data = dys[model_vars].dropna().copy()

model = smf.ols(formula, data=model_data).fit(
    cov_type="cluster", cov_kwds={"groups": model_data["uuid"]}
)

coef = model.params.get("reader_view", np.nan)
se = model.bse.get("reader_view", np.nan)
pval = model.pvalues.get("reader_view", np.nan)

# Convert log effect to percent change
pct_change = (np.exp(coef) - 1) * 100 if np.isfinite(coef) else np.nan

results = {
    "stats_basic": stats_basic,
    "t_test_raw_speed": t_raw,
    "cohen_d_raw_speed": float(d) if np.isfinite(d) else None,
    "regression": {
        "coef_reader_view_log_speed": float(coef) if np.isfinite(coef) else None,
        "se_reader_view_log_speed": float(se) if np.isfinite(se) else None,
        "pval_reader_view_log_speed": float(pval) if np.isfinite(pval) else None,
        "percent_change_speed": float(pct_change) if np.isfinite(pct_change) else None,
        "n_obs": int(model.nobs),
        "r2": float(model.rsquared),
    },
}

print(json.dumps(results, indent=2))
