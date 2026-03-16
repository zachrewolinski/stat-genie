import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

DATA_PATH = "panda_nuts.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Normalize categorical fields
for col in ["sex", "help", "hammer"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Efficiency: nuts opened per second
# Avoid division by zero if any
if (df["seconds"] <= 0).any():
    df = df[df["seconds"] > 0].copy()

df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Summary stats
summary = {
    "n_rows": int(len(df)),
    "efficiency_mean": float(df["efficiency"].mean()),
    "efficiency_std": float(df["efficiency"].std()),
}

# OLS on efficiency
ols_model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")

# Poisson on counts with offset for seconds
# Add small epsilon to avoid log(0) offset if seconds are zero (already filtered >0)
poisson_model = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"]),
).fit(cov_type="HC3")

# Null model for pseudo R^2
poisson_null = smf.glm(
    "nuts_opened ~ 1",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"]),
).fit()

# Collect p-values and coefficients
ols_results = ols_model.summary2().tables[1]
poisson_results = poisson_model.summary2().tables[1]

# Extract relevant terms
terms = ["age", "C(sex)[T.m]", "C(help)[T.y]"]

out = {
    "summary": summary,
    "ols": {
        "params": {t: float(ols_model.params.get(t, np.nan)) for t in terms},
        "pvalues": {t: float(ols_model.pvalues.get(t, np.nan)) for t in terms},
        "r2": float(ols_model.rsquared),
    },
    "poisson": {
        "params": {t: float(poisson_model.params.get(t, np.nan)) for t in terms},
        "pvalues": {t: float(poisson_model.pvalues.get(t, np.nan)) for t in terms},
        "pseudo_r2": float(1 - (poisson_model.llf / poisson_null.llf)),
    },
}

print(json.dumps(out, indent=2))
