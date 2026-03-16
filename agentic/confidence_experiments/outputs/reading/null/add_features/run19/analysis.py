import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
DF_PATH = "reading.csv"
df = pd.read_csv(DF_PATH)

# Ensure dyslexia_bin
if "dyslexia_bin" not in df.columns:
    if "dyslexia" in df.columns:
        df["dyslexia_bin"] = (df["dyslexia"] > 0).astype(int)
    else:
        raise ValueError("No dyslexia info")

# Filter dyslexic participants
sub = df[df["dyslexia_bin"] == 1].copy()

# Basic cleaning
cols_needed = ["speed", "reader_view", "uuid"]
for c in cols_needed:
    if c not in sub.columns:
        raise ValueError(f"Missing column: {c}")

sub = sub.replace([np.inf, -np.inf], np.nan)
sub = sub.dropna(subset=["speed", "reader_view", "uuid"])
sub = sub[sub["speed"] > 0]

# Descriptive stats by reader_view
summary = (
    sub.groupby("reader_view")["speed"]
    .agg(["count", "mean", "median", "std"])
    .reset_index()
)

# Welch t-test on raw speed
rv0 = sub[sub["reader_view"] == 0]["speed"]
rv1 = sub[sub["reader_view"] == 1]["speed"]

ttest = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")

# Mann-Whitney U test
try:
    mwu = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")
except Exception:
    mwu = None

# Effect size (Cohen's d, pooled SD)
# Use unbiased pooled SD
n1, n0 = rv1.shape[0], rv0.shape[0]
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)) if (n1 + n0 - 2) > 0 else np.nan
cohen_d = (rv1.mean() - rv0.mean()) / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan

# Regression with cluster-robust SE by uuid
sub["log_speed"] = np.log(sub["speed"])

# Build formula with available covariates
covariates = []
raw_covariates = []

for c in ["retake_trial"]:
    if c in sub.columns:
        covariates.append(c)
        raw_covariates.append(c)

for c in ["device", "page_id", "language"]:
    if c in sub.columns:
        covariates.append(f"C({c})")
        raw_covariates.append(c)

# Use readability/length if present and not collinear with page_id (keep if page_id absent)
if "num_words" in sub.columns and "page_id" not in sub.columns:
    covariates.append("num_words")
    raw_covariates.append("num_words")
if "Flesch_Kincaid" in sub.columns and "page_id" not in sub.columns:
    covariates.append("Flesch_Kincaid")
    raw_covariates.append("Flesch_Kincaid")

formula = "log_speed ~ reader_view"
if covariates:
    formula += " + " + " + ".join(covariates)

# Drop rows with missing covariates used in model
model_cols = ["log_speed", "reader_view", "uuid"] + raw_covariates
model_df = sub.dropna(subset=model_cols).copy()

model = smf.ols(formula, data=model_df).fit(
    cov_type="cluster",
    cov_kwds={"groups": model_df["uuid"]},
)

coef = model.params.get("reader_view", np.nan)
se = model.bse.get("reader_view", np.nan)
pval = model.pvalues.get("reader_view", np.nan)

# Convert log coefficient to % change
pct_change = (np.exp(coef) - 1) * 100 if pd.notnull(coef) else np.nan

results = {
    "n_total": int(sub.shape[0]),
    "n_reader_view_1": int(n1),
    "n_reader_view_0": int(n0),
    "summary_by_reader_view": summary.to_dict(orient="records"),
    "welch_ttest": {
        "tstat": float(ttest.statistic),
        "pvalue": float(ttest.pvalue),
    },
    "mannwhitney": None if mwu is None else {"stat": float(mwu.statistic), "pvalue": float(mwu.pvalue)},
    "cohen_d": float(cohen_d) if pd.notnull(cohen_d) else None,
    "regression": {
        "formula": formula,
        "coef_reader_view_log": float(coef) if pd.notnull(coef) else None,
        "se": float(se) if pd.notnull(se) else None,
        "pvalue": float(pval) if pd.notnull(pval) else None,
        "pct_change": float(pct_change) if pd.notnull(pct_change) else None,
        "r2": float(model.rsquared),
        "nobs": int(model.nobs),
    },
}

print(json.dumps(results, indent=2))
