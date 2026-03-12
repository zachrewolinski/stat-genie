import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = "teachingratings.csv"
df = pd.read_csv(csv_path)

# Basic cleaning
# Drop rows with missing values in key columns
key_cols = ["eval", "beauty"]
df_clean = df.dropna(subset=key_cols).copy()

# Pearson correlation
corr, corr_p = stats.pearsonr(df_clean["beauty"], df_clean["eval"])

# Simple OLS
model_simple = smf.ols("eval ~ beauty", data=df_clean).fit()

# Multivariate OLS with controls
# Convert categorical columns using C()
formula_controls = (
    "eval ~ beauty + age + students + allstudents + "
    "C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)"
)
model_controls = smf.ols(formula_controls, data=df_clean).fit()

# Collect results
results = {
    "n": int(df_clean.shape[0]),
    "corr": corr,
    "corr_p": corr_p,
    "simple_beta": model_simple.params.get("beauty", np.nan),
    "simple_p": model_simple.pvalues.get("beauty", np.nan),
    "simple_ci": model_simple.conf_int().loc["beauty"].tolist(),
    "controls_beta": model_controls.params.get("beauty", np.nan),
    "controls_p": model_controls.pvalues.get("beauty", np.nan),
    "controls_ci": model_controls.conf_int().loc["beauty"].tolist(),
    "simple_r2": model_simple.rsquared,
    "controls_r2": model_controls.rsquared,
}

print(json.dumps(results, indent=2))
