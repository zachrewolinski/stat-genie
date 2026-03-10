import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

DATA_PATH = "teachingratings.csv"

# Load data
_df = pd.read_csv(DATA_PATH)

# Core variables for the research question
core_cols = [
    "eval",
    "beauty",
    "age",
    "gender",
    "minority",
    "native",
    "tenure",
    "division",
    "credits",
    "students",
    "allstudents",
]

# Keep only columns that exist (defensive)
use_cols = [c for c in core_cols if c in _df.columns]

df = _df[use_cols].copy()

# Drop rows with missing values in used columns
n_before = len(df)
df = df.dropna()
n_after = len(df)

# Simple correlation
corr = df["beauty"].corr(df["eval"])

# OLS: unadjusted
model_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

# OLS: adjusted with controls
# Use categorical encoding for discrete factors
formula_parts = ["beauty"]
for cat in ["gender", "minority", "native", "tenure", "division", "credits"]:
    if cat in df.columns:
        formula_parts.append(f"C({cat})")
for num in ["age", "students", "allstudents"]:
    if num in df.columns:
        formula_parts.append(num)
formula = "eval ~ " + " + ".join(formula_parts)

model_adj = smf.ols(formula, data=df).fit(cov_type="HC3")

# Standardized coefficient for beauty in adjusted model
# z-score beauty and eval to get standardized beta
zdf = df.copy()
for col in ["beauty", "eval"]:
    zdf[col] = (zdf[col] - zdf[col].mean()) / zdf[col].std(ddof=0)

z_formula = "eval ~ " + " + ".join(formula_parts)
model_adj_z = smf.ols(z_formula, data=zdf).fit(cov_type="HC3")

results = {
    "n_before": n_before,
    "n_after": n_after,
    "corr": corr,
    "simple_coef": model_simple.params.get("beauty", np.nan),
    "simple_p": model_simple.pvalues.get("beauty", np.nan),
    "simple_ci": list(model_simple.conf_int().loc["beauty"].values),
    "adj_coef": model_adj.params.get("beauty", np.nan),
    "adj_p": model_adj.pvalues.get("beauty", np.nan),
    "adj_ci": list(model_adj.conf_int().loc["beauty"].values),
    "adj_r2": model_adj.rsquared,
    "adj_beta_std": model_adj_z.params.get("beauty", np.nan),
}

print(json.dumps(results, indent=2))
