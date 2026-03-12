import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "reading.csv"

# Load data
# Use low_memory=False to avoid dtype warnings for mixed types
_df = pd.read_csv(DATA_PATH, low_memory=False)

# Focus on dyslexia group
if "dyslexia_bin" not in _df.columns:
    raise ValueError("Expected dyslexia_bin column not found")

# Basic cleaning
_df = _df.copy()
_df = _df.replace([np.inf, -np.inf], np.nan)

# Filter dyslexia group (includes severe dyslexia as well)
_df_dys = _df[_df["dyslexia_bin"] == 1].copy()

# Keep valid speed values
_df_dys = _df_dys[_df_dys["speed"].notna()]
_df_dys = _df_dys[_df_dys["speed"] > 0]

# Ensure reader_view is present
_df_dys = _df_dys[_df_dys["reader_view"].notna()]

# Group stats
_group_stats = (
    _df_dys.groupby("reader_view")["speed"]
    .agg(["count", "mean", "median", "std"])
    .rename(index={0: "off", 1: "on"})
)

# Log speed for more stable inference
_df_dys["log_speed"] = np.log(_df_dys["speed"])

# Two-sample Welch t-test on log speed
rv1 = _df_dys[_df_dys["reader_view"] == 1]["log_speed"]
rv0 = _df_dys[_df_dys["reader_view"] == 0]["log_speed"]

ttest = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")

# Effect size (Cohen's d) on log speed
n1 = rv1.shape[0]
n0 = rv0.shape[0]
m1 = rv1.mean()
m0 = rv0.mean()
var1 = rv1.var(ddof=1)
var0 = rv0.var(ddof=1)
pooled_std = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)) if (n1 + n0 - 2) > 0 else np.nan
cohen_d = (m1 - m0) / pooled_std if pooled_std and pooled_std > 0 else np.nan

# Percent change in geometric mean speed
log_diff = m1 - m0
pct_change = float(np.exp(log_diff) - 1)

# Mann-Whitney U test on speed (non-parametric)
try:
    mwu = stats.mannwhitneyu(
        _df_dys[_df_dys["reader_view"] == 1]["speed"],
        _df_dys[_df_dys["reader_view"] == 0]["speed"],
        alternative="two-sided",
    )
except ValueError:
    mwu = None

# Regression with covariates and cluster-robust SEs by participant
# Choose a conservative set of covariates to avoid multicollinearity
covariates = [
    "reader_view",
    "num_words",
    "Flesch_Kincaid",
    "age",
    "gender",
    "retake_trial",
]
cat_vars = ["device", "page_id", "language", "english_native"]

# Build regression frame
reg_cols = ["uuid", "log_speed"] + covariates + cat_vars
reg_df = _df_dys[reg_cols].dropna().copy()

# One-hot encode categoricals
reg_df = pd.get_dummies(reg_df, columns=cat_vars, drop_first=True)

X = reg_df.drop(columns=["log_speed", "uuid"])
X = sm.add_constant(X, has_constant="add")
y = reg_df["log_speed"]

# Fit OLS with cluster-robust SEs
model = sm.OLS(y, X)
res = model.fit(cov_type="cluster", cov_kwds={"groups": reg_df["uuid"]})

coef = res.params.get("reader_view", np.nan)
se = res.bse.get("reader_view", np.nan)
pval = res.pvalues.get("reader_view", np.nan)

pct_change_adj = float(np.exp(coef) - 1) if np.isfinite(coef) else np.nan

# Prepare results
results = {
    "n_total_dyslexia": int(_df_dys.shape[0]),
    "group_stats": _group_stats.reset_index().rename(columns={"reader_view": "reader_view_state"}).to_dict(orient="records"),
    "ttest_log_speed": {
        "t_stat": float(ttest.statistic),
        "p_value": float(ttest.pvalue),
        "log_mean_diff": float(log_diff),
        "pct_change_geo_mean": pct_change,
        "cohen_d": float(cohen_d),
        "n_reader_view_on": int(n1),
        "n_reader_view_off": int(n0),
    },
    "mannwhitney_speed": None if mwu is None else {
        "u_stat": float(mwu.statistic),
        "p_value": float(mwu.pvalue),
    },
    "regression_log_speed_clustered": {
        "coef_reader_view": float(coef),
        "se": float(se),
        "p_value": float(pval),
        "pct_change_adj": pct_change_adj,
        "n_used": int(reg_df.shape[0]),
        "num_predictors": int(X.shape[1]),
    },
}

print(json.dumps(results, indent=2))
