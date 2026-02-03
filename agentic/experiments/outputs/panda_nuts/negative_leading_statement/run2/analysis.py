import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv("panda_nuts.csv")

# Basic cleaning/typing
for col in ["sex", "help", "hammer"]:
    df[col] = df[col].astype("category")

# Efficiency as rate (nuts per second)
df["rate"] = df["nuts_opened"] / df["seconds"]

# GLM Poisson with log(seconds) offset to model rate properly
# Predictors: age (continuous), sex, help
formula = "nuts_opened ~ age + sex + help"
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["seconds"])
).fit(cov_type="HC3")

print("GLM Poisson with offset (nuts_opened ~ age + sex + help, exposure=seconds)")
print(model.summary())

# OLS on rate as a simple robustness check
ols = smf.ols("rate ~ age + sex + help", data=df).fit(cov_type="HC3")
print("\nOLS on rate (nuts_opened/seconds)")
print(ols.summary())

# Group summaries
summary = (
    df.groupby(["sex", "help"], observed=True)["rate"]
    .agg(["mean", "std", "count"])\
    .reset_index()
)
print("\nRate summary by sex and help")
print(summary.to_string(index=False))
