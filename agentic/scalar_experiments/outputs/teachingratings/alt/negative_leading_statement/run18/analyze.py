import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = "teachingratings.csv"
df = pd.read_csv(csv_path)

# Basic cleaning: ensure expected columns
# Convert categorical columns to category dtype
cat_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Compute simple correlation between beauty and eval
corr, corr_p = stats.pearsonr(df["beauty"], df["eval"])

# Simple OLS: eval ~ beauty
model_simple = smf.ols("eval ~ beauty", data=df).fit()

# Create log versions of class size variables to reduce skew
for col in ["students", "allstudents"]:
    if col in df.columns:
        df[f"log_{col}"] = np.log1p(df[col])

# Controlled OLS with typical covariates
# Use log_students and log_allstudents when available
formula_terms = [
    "beauty",
    "age",
    "C(gender)",
    "C(minority)",
    "C(native)",
    "C(tenure)",
    "C(division)",
    "C(credits)",
]
if "log_students" in df.columns:
    formula_terms.append("log_students")
if "log_allstudents" in df.columns:
    formula_terms.append("log_allstudents")

formula = "eval ~ " + " + ".join(formula_terms)
model_controls = smf.ols(formula, data=df).fit()

# Standardized effect of beauty on eval for controlled model
# Standardize using sample SDs
beauty_sd = df["beauty"].std(ddof=1)
eval_sd = df["eval"].std(ddof=1)

beauty_coef_ctrl = model_controls.params.get("beauty", np.nan)
beauty_se_ctrl = model_controls.bse.get("beauty", np.nan)
beauty_p_ctrl = model_controls.pvalues.get("beauty", np.nan)

# Effect of 1 SD beauty on eval in SD units
std_effect_ctrl = (beauty_coef_ctrl * beauty_sd) / eval_sd if eval_sd != 0 else np.nan

# Gather key stats
results = {
    "n": int(model_simple.nobs),
    "corr": corr,
    "corr_p": corr_p,
    "simple_coef": model_simple.params["beauty"],
    "simple_p": model_simple.pvalues["beauty"],
    "simple_r2": model_simple.rsquared,
    "ctrl_coef": beauty_coef_ctrl,
    "ctrl_se": beauty_se_ctrl,
    "ctrl_p": beauty_p_ctrl,
    "ctrl_r2": model_controls.rsquared,
    "std_effect_ctrl": std_effect_ctrl,
}

print(json.dumps(results, indent=2))
