import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Efficiency: nuts opened per second
# (Avoid division by zero; seconds min > 0 per metadata)
df = df.copy()
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Ensure categorical types
for col in ["sex", "help"]:
    df[col] = df[col].astype("category")

# Basic descriptives
print("Rows:", len(df))
print("Efficiency summary:\n", df["efficiency"].describe())
print("Mean efficiency by sex:\n", df.groupby("sex")["efficiency"].mean())
print("Mean efficiency by help:\n", df.groupby("help")["efficiency"].mean())

# OLS on efficiency with cluster-robust SE by chimpanzee
ols = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]}
)
print("\nOLS (clustered by chimpanzee) on efficiency")
print(ols.summary())

# Poisson rate model on nuts opened with offset for time
# This models nut-cracking rate per second.
df["log_seconds"] = np.log(df["seconds"])
pois = smf.glm(
    "nuts_opened ~ age + C(sex) + C(help)",
    data=df,
    family=sm.families.Poisson(),
    offset=df["log_seconds"],
).fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})

print("\nPoisson rate model (clustered by chimpanzee)")
print(pois.summary())
print("Rate ratios:\n", np.exp(pois.params))
