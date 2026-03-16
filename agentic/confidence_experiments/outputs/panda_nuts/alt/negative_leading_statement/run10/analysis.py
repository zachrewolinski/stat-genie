import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Basic cleaning
# Standardize categorical values
for col in ["sex", "help", "hammer"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Efficiency: nuts opened per second
# Avoid division by zero
if (df["seconds"] <= 0).any():
    raise ValueError("Non-positive seconds found")

df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Convert categorical variables
for col in ["sex", "help"]:
    df[col] = df[col].astype("category")

# Descriptives
summary = {
    "n_rows": len(df),
    "n_chimpanzees": df["chimpanzee"].nunique(),
    "efficiency_mean": df["efficiency"].mean(),
    "efficiency_std": df["efficiency"].std(),
    "efficiency_min": df["efficiency"].min(),
    "efficiency_max": df["efficiency"].max(),
    "sex_counts": df["sex"].value_counts(dropna=False).to_dict(),
    "help_counts": df["help"].value_counts(dropna=False).to_dict(),
}

# Mixed effects model: random intercept for chimpanzee
# Use reml=False for ML
model_mixed = smf.mixedlm("efficiency ~ age + sex + help", df, groups=df["chimpanzee"])
try:
    res_mixed = model_mixed.fit(reml=False)
except Exception as e:
    # Fallback to OLS if mixed fails
    res_mixed = None

# OLS with cluster-robust SE by chimpanzee
ols = smf.ols("efficiency ~ age + sex + help", df).fit(
    cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]}
)

# Poisson GLM on counts with log(seconds) offset for rate
# Add small constant if zeros? Poisson handles zeros in response.
# Use robust SE clustered by chimpanzee via GEE? We'll use GLM with cluster-robust SE.
poisson = smf.glm(
    "nuts_opened ~ age + sex + help",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"])
).fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})

# Collect coefficients and p-values

def extract(res):
    return {
        "params": res.params.to_dict(),
        "pvalues": res.pvalues.to_dict(),
        "bse": res.bse.to_dict(),
        "nobs": int(res.nobs),
    }

results = {
    "summary": summary,
    "mixed": None,
    "ols_cluster": extract(ols),
    "poisson_cluster": extract(poisson),
}
if res_mixed is not None:
    results["mixed"] = extract(res_mixed)

print(json.dumps(results, indent=2))
