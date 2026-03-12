import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv("reading.csv")

# Define dyslexia subset: dyslexia_bin == 1 (includes dyslexia and severe dyslexia)
# Fallback to dyslexia > 0 if dyslexia_bin missing
if "dyslexia_bin" in _df.columns:
    dys_df = _df[_df["dyslexia_bin"] == 1].copy()
else:
    dys_df = _df[_df["dyslexia"] > 0].copy()

# Basic cleaning: remove non-positive or missing speed
speed_col = "speed"

dys_df = dys_df.replace([np.inf, -np.inf], np.nan)

dys_df = dys_df[dys_df[speed_col].notna()]
# keep strictly positive speeds for log transform

dys_df = dys_df[dys_df[speed_col] > 0]

# Ensure reader_view binary
rv = dys_df["reader_view"]

# Descriptive stats by reader_view
summary = (
    dys_df.groupby("reader_view")[speed_col]
    .agg(["count", "mean", "median", "std"])
    .rename_axis("reader_view")
)

# Welch t-test on raw speed
speed_rv1 = dys_df.loc[dys_df["reader_view"] == 1, speed_col]
speed_rv0 = dys_df.loc[dys_df["reader_view"] == 0, speed_col]

# Guard: need at least 2 values per group
if len(speed_rv1) >= 2 and len(speed_rv0) >= 2:
    ttest = stats.ttest_ind(speed_rv1, speed_rv0, equal_var=False, nan_policy="omit")
else:
    ttest = None

# Effect size: Cohen's d (Welch)
# Use pooled SD with unequal sizes
if len(speed_rv1) >= 2 and len(speed_rv0) >= 2:
    n1, n0 = len(speed_rv1), len(speed_rv0)
    s1, s0 = speed_rv1.std(ddof=1), speed_rv0.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
    cohend = (speed_rv1.mean() - speed_rv0.mean()) / sp if sp > 0 else np.nan
else:
    cohend = np.nan

# Regression with cluster-robust SE by uuid
# Use log(speed) to reduce skew

dys_df["log_speed"] = np.log(dys_df[speed_col])

# Build model with covariates likely relevant
covariates = [
    "reader_view",
    "num_words",
    "Flesch_Kincaid",
    "img_width",
    "scrolling_time",
    "age",
    "retake_trial",
]

# Include categorical covariates where available
cat_covars = ["device", "language", "education", "gender", "english_native"]

# Keep only columns that exist and have variability
use_covars = [c for c in covariates if c in dys_df.columns]

# Prepare design matrix
X = dys_df[use_covars].copy()

# Add categorical dummies
for c in cat_covars:
    if c in dys_df.columns:
        # Skip if only one category in dyslexia subset
        if dys_df[c].nunique(dropna=True) > 1:
            dummies = pd.get_dummies(dys_df[c], prefix=c, drop_first=True)
            X = pd.concat([X, dummies], axis=1)

# Drop rows with missing values in X or y
model_df = pd.concat([dys_df["log_speed"], X, dys_df[["uuid"]]], axis=1)
model_df = model_df.dropna()

# Ensure reader_view is numeric
if "reader_view" in model_df.columns:
    model_df["reader_view"] = pd.to_numeric(model_df["reader_view"], errors="coerce")

# Fit OLS with cluster-robust SE
if "reader_view" in model_df.columns and model_df["reader_view"].nunique() > 1:
    y = model_df["log_speed"]
    X_model = sm.add_constant(model_df.drop(columns=["log_speed", "uuid"]))
    ols_model = sm.OLS(y, X_model).fit(cov_type="cluster", cov_kwds={"groups": model_df["uuid"]})
    rv_coef = ols_model.params.get("reader_view", np.nan)
    rv_pval = ols_model.pvalues.get("reader_view", np.nan)
    rv_ci = ols_model.conf_int().loc["reader_view"].tolist() if "reader_view" in ols_model.params else [np.nan, np.nan]
else:
    ols_model = None
    rv_coef = np.nan
    rv_pval = np.nan
    rv_ci = [np.nan, np.nan]

# Convert log-speed coef to percent change
if rv_coef == rv_coef:  # not nan
    pct_change = (np.exp(rv_coef) - 1) * 100
else:
    pct_change = np.nan

# Compile results for quick inspection
results = {
    "dyslexia_n": int(dys_df.shape[0]),
    "reader_view_counts": dys_df["reader_view"].value_counts(dropna=False).to_dict(),
    "summary_by_reader_view": summary.reset_index().to_dict(orient="records"),
    "ttest": {
        "t_stat": float(ttest.statistic) if ttest else None,
        "p_value": float(ttest.pvalue) if ttest else None,
        "cohens_d": float(cohend) if cohend == cohend else None,
    },
    "regression": {
        "rv_coef_log": float(rv_coef) if rv_coef == rv_coef else None,
        "rv_p_value": float(rv_pval) if rv_pval == rv_pval else None,
        "rv_ci_log": [float(rv_ci[0]), float(rv_ci[1])] if rv_ci[0] == rv_ci[0] else [None, None],
        "rv_pct_change": float(pct_change) if pct_change == pct_change else None,
        "n_used": int(model_df.shape[0]),
        "n_clusters": int(model_df["uuid"].nunique()) if "uuid" in model_df.columns else None,
    },
}

print(json.dumps(results, indent=2))
