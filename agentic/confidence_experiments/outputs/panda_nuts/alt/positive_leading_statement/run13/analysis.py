import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = "panda_nuts.csv"
df = pd.read_csv(csv_path)

# Compute efficiency: nuts opened per second
# Avoid division by zero if any seconds == 0
if (df["seconds"] <= 0).any():
    raise ValueError("Non-positive session duration found.")

df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Basic descriptive stats
n_rows = len(df)
unique_chimps = df["chimpanzee"].nunique()

# OLS with cluster-robust SE by chimpanzee to account for repeated measures
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]}
)

# Extract coefficients and p-values
params = model.params
pvalues = model.pvalues
conf = model.conf_int()

results = {
    "n_rows": int(n_rows),
    "unique_chimps": int(unique_chimps),
    "r2": float(model.rsquared),
    "params": {k: float(v) for k, v in params.items()},
    "pvalues": {k: float(v) for k, v in pvalues.items()},
    "conf_int": {k: [float(conf.loc[k, 0]), float(conf.loc[k, 1])] for k in conf.index},
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
