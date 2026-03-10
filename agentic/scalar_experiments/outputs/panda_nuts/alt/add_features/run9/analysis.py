import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv("panda_nuts.csv")

# Basic cleaning: keep rows with required fields
needed = ["age", "sex", "help", "nuts_opened", "seconds"]
clean = df.dropna(subset=needed).copy()

# Compute efficiency: nuts opened per second
clean["efficiency"] = clean["nuts_opened"] / clean["seconds"]

# Ensure categories
clean["sex"] = clean["sex"].astype(str)
clean["help"] = clean["help"].astype(str)

# OLS with categorical predictors
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=clean).fit()

# Also try log efficiency to reduce skew (add small constant)
clean["log_eff"] = np.log(clean["efficiency"] + 1e-6)
log_model = smf.ols("log_eff ~ age + C(sex) + C(help)", data=clean).fit()

# Summaries
summary = {
    "n": int(len(clean)),
    "efficiency_mean": float(clean["efficiency"].mean()),
    "efficiency_std": float(clean["efficiency"].std()),
    "ols_params": model.params.to_dict(),
    "ols_pvalues": model.pvalues.to_dict(),
    "ols_r2": float(model.rsquared),
    "log_params": log_model.params.to_dict(),
    "log_pvalues": log_model.pvalues.to_dict(),
    "log_r2": float(log_model.rsquared),
}

# Group means for sex/help
summary["means_by_sex"] = clean.groupby("sex")["efficiency"].mean().to_dict()
summary["means_by_help"] = clean.groupby("help")["efficiency"].mean().to_dict()

print(json.dumps(summary, indent=2, sort_keys=True))
