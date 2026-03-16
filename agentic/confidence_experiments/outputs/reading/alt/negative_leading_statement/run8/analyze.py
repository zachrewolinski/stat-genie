import json
import math
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "reading.csv"

df = pd.read_csv(DATA_PATH)

# Focus on participants with dyslexia (binary flag)
if "dyslexia_bin" in df.columns:
    dys_df = df[df["dyslexia_bin"] == 1].copy()
else:
    dys_df = df[df["dyslexia"].fillna(0) > 0].copy()

# Basic cleaning
cols_needed = ["speed", "reader_view", "uuid", "page_id", "device", "age", "gender", "english_native"]
for c in cols_needed:
    if c not in dys_df.columns:
        dys_df[c] = np.nan

dys_df = dys_df.dropna(subset=["speed", "reader_view", "uuid"]).copy()

# Remove non-positive speeds for log transform

dys_df = dys_df[dys_df["speed"] > 0].copy()

dys_df["log_speed"] = np.log(dys_df["speed"])

# Descriptive stats by reader_view
summary = dys_df.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).to_dict()

# Cohen's d (independent groups)
rv1 = dys_df[dys_df["reader_view"] == 1]["speed"].values
rv0 = dys_df[dys_df["reader_view"] == 0]["speed"].values

def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    return (np.mean(a) - np.mean(b)) / math.sqrt(pooled) if pooled > 0 else float("nan")

cohens_d_val = cohens_d(rv1, rv0)

# Two-sample t-test (Welch)
try:
    ttest = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
    ttest_res = {"statistic": float(ttest.statistic), "pvalue": float(ttest.pvalue)}
except Exception:
    ttest_res = {"statistic": None, "pvalue": None}

# Paired within-subject analysis (only participants with both conditions)
paired = (
    dys_df.groupby(["uuid", "reader_view"])["speed"]
    .mean()
    .unstack("reader_view")
)
if 0 in paired.columns and 1 in paired.columns:
    paired = paired.dropna(subset=[0, 1])
    paired["diff"] = paired[1] - paired[0]
    if len(paired) >= 2:
        ttest_paired = stats.ttest_1samp(paired["diff"], 0.0, nan_policy="omit")
        paired_res = {
            "n": int(len(paired)),
            "mean_diff": float(paired["diff"].mean()),
            "median_diff": float(paired["diff"].median()),
            "statistic": float(ttest_paired.statistic),
            "pvalue": float(ttest_paired.pvalue),
        }
    else:
        paired_res = {"n": int(len(paired)), "mean_diff": None, "median_diff": None, "statistic": None, "pvalue": None}
else:
    paired_res = {"n": 0, "mean_diff": None, "median_diff": None, "statistic": None, "pvalue": None}

# Regression with cluster-robust SE by participant
# Keep rows with necessary covariates (some may be missing)
model_df = dys_df.dropna(subset=["page_id", "device", "age", "gender", "english_native"]).copy()

# Ensure categorical types
for c in ["page_id", "device", "english_native"]:
    model_df[c] = model_df[c].astype("category")

# Use log_speed to reduce skew; control for page and device; age/gender/native
# reader_view is 0/1 numeric
formula = "log_speed ~ reader_view + C(page_id) + C(device) + age + gender + C(english_native)"

reg_res = None
if len(model_df) >= 30:
    model = smf.ols(formula, data=model_df)
    reg_res = model.fit(cov_type="cluster", cov_kwds={"groups": model_df["uuid"]})

    coef = float(reg_res.params.get("reader_view", float("nan")))
    se = float(reg_res.bse.get("reader_view", float("nan")))
    tval = float(reg_res.tvalues.get("reader_view", float("nan")))
    pval = float(reg_res.pvalues.get("reader_view", float("nan")))
    # 95% CI
    ci_low, ci_high = reg_res.conf_int().loc["reader_view"].tolist()
    # percent change in speed
    pct_change = (math.exp(coef) - 1.0) * 100.0

    reg_summary = {
        "n": int(reg_res.nobs),
        "coef_log": coef,
        "se": se,
        "t": tval,
        "p": pval,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "pct_change": pct_change,
    }
else:
    reg_summary = None

results = {
    "n_dyslexic_obs": int(len(dys_df)),
    "n_dyslexic_participants": int(dys_df["uuid"].nunique()),
    "summary_speed_by_reader_view": summary,
    "cohens_d": cohens_d_val,
    "ttest": ttest_res,
    "paired": paired_res,
    "regression": reg_summary,
}

print(json.dumps(results, indent=2))

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
