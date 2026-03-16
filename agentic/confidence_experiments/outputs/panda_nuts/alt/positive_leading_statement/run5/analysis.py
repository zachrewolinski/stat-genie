import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("panda_nuts.csv")

# Compute efficiency: nuts opened per second
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Basic sanity checks
print("rows", len(df))
print(df[["efficiency", "nuts_opened", "seconds"]].describe())

# Fit OLS with cluster-robust SE by chimpanzee (repeated measures)
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]}
)
print(model.summary())

# Also compute plain OLS for reference
model_plain = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit()
print("\nPlain OLS (non-robust)")
print(model_plain.summary())

# Extract key results
print("\nCluster-robust coefficients:")
print(model.params)
print(model.pvalues)
