import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure expected columns exist
required_cols = ["age", "sex", "help", "nuts_opened", "seconds"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Drop rows with missing values in required columns
sub = df[required_cols].copy()
sub = sub.dropna()

# Compute efficiency: nuts opened per second
sub["efficiency"] = sub["nuts_opened"] / sub["seconds"]

# Fit OLS model with categorical predictors
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=sub).fit(cov_type="HC3")

# Also fit Poisson with offset (nuts opened count) as a check
# Avoid log(0) by excluding zero seconds, already handled by seconds > 0
sub = sub[sub["seconds"] > 0].copy()
sub["log_seconds"] = np.log(sub["seconds"])
poisson_model = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=sub,
    family=sm.families.Poisson(),
    offset=sub["log_seconds"],
).fit(cov_type="HC3")

# Summaries
results = {
    "n_rows": int(len(sub)),
    "efficiency_mean": float(sub["efficiency"].mean()),
    "efficiency_std": float(sub["efficiency"].std()),
    "ols_params": model.params.to_dict(),
    "ols_pvalues": model.pvalues.to_dict(),
    "ols_r2": float(model.rsquared),
    "poisson_params": poisson_model.params.to_dict(),
    "poisson_pvalues": poisson_model.pvalues.to_dict(),
}

print(json.dumps(results, indent=2))
