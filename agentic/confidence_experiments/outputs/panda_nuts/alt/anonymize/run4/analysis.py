import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

csv_path = "panda_nuts.csv"

df = pd.read_csv(csv_path)

# Rename for clarity
col_map = {
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer",
    "feature5": "nuts_opened",
    "feature6": "duration_sec",
    "feature7": "help",
}

df = df.rename(columns=col_map)

# Compute efficiency: nuts opened per second
# Avoid division by zero (none expected per metadata)
df["efficiency"] = df["nuts_opened"] / df["duration_sec"]

# Clean categorical variables
# Ensure consistent categories

df["sex"] = df["sex"].astype("category")
# help appears to be 'y' and 'N' per metadata
# Normalize to lowercase y/n for clarity

df["help"] = df["help"].astype(str).str.lower().map({"y": "y", "n": "n"})
df["help"] = df["help"].astype("category")

# Basic stats
summary = {
    "n_rows": int(df.shape[0]),
    "efficiency_mean": float(df["efficiency"].mean()),
    "efficiency_std": float(df["efficiency"].std()),
    "efficiency_min": float(df["efficiency"].min()),
    "efficiency_max": float(df["efficiency"].max()),
    "sex_counts": df["sex"].value_counts(dropna=False).to_dict(),
    "help_counts": df["help"].value_counts(dropna=False).to_dict(),
}

# OLS model on raw efficiency
model_raw = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit()

# Log-transform to mitigate skew: log(1 + efficiency)
# (efficiency is non-negative)

df["log_efficiency"] = np.log1p(df["efficiency"])
model_log = smf.ols("log_efficiency ~ age + C(sex) + C(help)", data=df).fit()

# ANOVA (type II) for overall effect of each predictor
anova_raw = anova_lm(model_raw, typ=2)
anova_log = anova_lm(model_log, typ=2)

# Export key results
results = {
    "summary": summary,
    "raw_model": {
        "params": model_raw.params.to_dict(),
        "pvalues": model_raw.pvalues.to_dict(),
        "r2": float(model_raw.rsquared),
        "adj_r2": float(model_raw.rsquared_adj),
        "anova": anova_raw[["F", "PR(>F)"]].to_dict(orient="index"),
    },
    "log_model": {
        "params": model_log.params.to_dict(),
        "pvalues": model_log.pvalues.to_dict(),
        "r2": float(model_log.rsquared),
        "adj_r2": float(model_log.rsquared_adj),
        "anova": anova_log[["F", "PR(>F)"]].to_dict(orient="index"),
    },
}

print(json.dumps(results, indent=2, sort_keys=True))
