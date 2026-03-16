import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv("panda_nuts.csv")

# Efficiency: nuts opened per second
# Guard against division by zero just in case
if (df["feature6"] <= 0).any():
    raise ValueError("Non-positive session durations found; cannot compute efficiency.")

df["efficiency"] = df["feature5"] / df["feature6"]

# Basic summaries
summary = {
    "n": int(df.shape[0]),
    "efficiency_mean": float(df["efficiency"].mean()),
    "efficiency_std": float(df["efficiency"].std(ddof=1)),
    "efficiency_min": float(df["efficiency"].min()),
    "efficiency_max": float(df["efficiency"].max()),
}

# Group means for interpretation
sex_means = df.groupby("feature3")["efficiency"].mean().to_dict()
help_means = df.groupby("feature7")["efficiency"].mean().to_dict()

# OLS with robust (HC3) standard errors
model = smf.ols("efficiency ~ feature2 + C(feature3) + C(feature7)", data=df).fit(cov_type="HC3")

coef_table = {}
for name in model.params.index:
    coef_table[name] = {
        "coef": float(model.params[name]),
        "se": float(model.bse[name]),
        "p": float(model.pvalues[name]),
    }

results = {
    "summary": summary,
    "sex_means": sex_means,
    "help_means": help_means,
    "r2": float(model.rsquared),
    "adj_r2": float(model.rsquared_adj),
    "nobs": int(model.nobs),
    "params": {k: float(v) for k, v in model.params.items()},
    "pvalues": {k: float(v) for k, v in model.pvalues.items()},
    "coef_table": coef_table,
}

print(json.dumps(results, indent=2))
