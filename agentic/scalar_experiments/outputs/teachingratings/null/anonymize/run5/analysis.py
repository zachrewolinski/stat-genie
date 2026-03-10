import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_PATH = "teachingratings.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Map columns
beauty_col = "feature6"
rating_col = "feature7"
cluster_col = "feature13"

# Basic cleaning
_df = _df.copy()

# Drop rows with missing key variables
_df = _df.dropna(subset=[beauty_col, rating_col])

n = len(_df)

# Pearson correlation
corr, corr_p = stats.pearsonr(_df[beauty_col], _df[rating_col])

# Simple OLS
X_simple = sm.add_constant(_df[[beauty_col]])
ols_simple = sm.OLS(_df[rating_col], X_simple).fit()

# Cluster-robust SE by instructor id
ols_simple_cl = ols_simple.get_robustcov_results(cov_type="cluster", groups=_df[cluster_col])

# Multiple regression with controls
# Categorical columns
cat_cols = ["feature2", "feature4", "feature5", "feature8", "feature9", "feature10"]
num_cols = ["feature3", "feature11", "feature12"]

# Build design matrix
X = _df[[beauty_col] + num_cols + cat_cols].copy()

# Convert categoricals to dummies
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X = sm.add_constant(X)
ols_multi = sm.OLS(_df[rating_col], X).fit()
ols_multi_cl = ols_multi.get_robustcov_results(cov_type="cluster", groups=_df[cluster_col])

# Effect sizes
beauty_sd = _df[beauty_col].std(ddof=1)
coef_simple = ols_simple.params[beauty_col]
coef_multi = ols_multi.params[beauty_col]

# Per 1 SD increase in beauty
effect_simple_1sd = coef_simple * beauty_sd
effect_multi_1sd = coef_multi * beauty_sd

# Prepare results summary for later use
results = {
    "n": int(n),
    "corr": float(corr),
    "corr_p": float(corr_p),
    "simple": {
        "coef": float(coef_simple),
        "se": float(ols_simple.bse[beauty_col]),
        "p": float(ols_simple.pvalues[beauty_col]),
        "r2": float(ols_simple.rsquared),
        "coef_cluster": float(ols_simple_cl.params[1]),
        "se_cluster": float(ols_simple_cl.bse[1]),
        "p_cluster": float(ols_simple_cl.pvalues[1]),
    },
    "multi": {
        "coef": float(coef_multi),
        "se": float(ols_multi.bse[beauty_col]),
        "p": float(ols_multi.pvalues[beauty_col]),
        "r2": float(ols_multi.rsquared),
        "coef_cluster": float(ols_multi_cl.params[list(ols_multi.params.index).index(beauty_col)]),
        "se_cluster": float(ols_multi_cl.bse[list(ols_multi.params.index).index(beauty_col)]),
        "p_cluster": float(ols_multi_cl.pvalues[list(ols_multi.params.index).index(beauty_col)]),
    },
    "beauty_sd": float(beauty_sd),
    "effect_simple_1sd": float(effect_simple_1sd),
    "effect_multi_1sd": float(effect_multi_1sd),
}

print(json.dumps(results, indent=2))
