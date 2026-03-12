import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)
# Map columns
id_col = "feature1"
age_col = "feature2"
sex_col = "feature3"
hammer_col = "feature4"
nuts_col = "feature5"
secs_col = "feature6"
help_col = "feature7"

# Efficiency: nuts per second
# Avoid division by zero (shouldn't exist) by filtering

df = df.copy()

df["efficiency"] = df[nuts_col] / df[secs_col]

# Clean categorical labels (help has 'y' and 'N')
# Ensure consistent casing

df[sex_col] = df[sex_col].astype(str).str.lower()
df[help_col] = df[help_col].astype(str).str.lower()

# Basic summaries
summary = {
    "n_rows": int(df.shape[0]),
    "efficiency_mean": float(df["efficiency"].mean()),
    "efficiency_std": float(df["efficiency"].std()),
    "efficiency_min": float(df["efficiency"].min()),
    "efficiency_max": float(df["efficiency"].max()),
    "age_mean": float(df[age_col].mean()),
}

# Group means
sex_means = df.groupby(sex_col)["efficiency"].agg(["mean", "std", "count"]).to_dict("index")
help_means = df.groupby(help_col)["efficiency"].agg(["mean", "std", "count"]).to_dict("index")

# OLS with robust SE
# Treat sex and help as categorical
model = smf.ols("efficiency ~ feature2 + C(feature3) + C(feature7)", data=df).fit(cov_type="HC3")

results = {
    "summary": summary,
    "sex_means": sex_means,
    "help_means": help_means,
    "params": model.params.to_dict(),
    "pvalues": model.pvalues.to_dict(),
    "conf_int": {k: list(v) for k, v in model.conf_int().to_dict("index").items()},
    "r2": float(model.rsquared),
    "adj_r2": float(model.rsquared_adj),
}

print(json.dumps(results, indent=2, sort_keys=True))
